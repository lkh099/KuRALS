"""NPU feasibility check for any model in this repo: derive its layer list, emit a cfg_gen
input via manifest_to_cfg_input, and run npu_riscv's NPU8 gen_cfg.py on it (PF=8).

Reports pass/fail of the generator's buffer allocation at a given per-buffer size, the buffer
requirement it computed, total param bytes, plus MACs/params/peak activation from the layer list.
  python -m kurals.npu_check --model npu_unet --width-mult 1.0 --height 32 --width 64
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

from kurals.manifest_to_cfg_input import manifest_to_cfg
from kurals.models import KuRALSNetNPUSeg, NPUUNet
from kurals.models.quant import QuantConvBNAct

GEN_CFG_DIR = os.environ.get('NPU_RISCV_CFG_DIR', os.path.expanduser('~/xsrc/ext/npu_riscv/cfg_generator'))


def _entry(name, m, **fusion):
    c = m.conv
    e = dict(name=name, kernel_size=c.kernel_size[0], stride=m.stride, in_ch=c.in_channels,
             out_ch=c.out_channels, depthwise=m.groups > 1, act_name=m.act_name, shift1=12, shift2=3)
    e.update(fusion)
    return e


def topology(net):
    """Ordered conv entries (manifest-style) for an NPU model."""
    if isinstance(net, NPUUNet):
        out = []
        for s in net.specs:
            f = {k: s[k] for k in ('route_source', 'upsample_after', 'maxpool_after') if k in s}
            out.append(_entry(s['name'], net.convs[s['name']], **f))
        return out
    assert isinstance(net, KuRALSNetNPUSeg), type(net)
    ed = net.encoder_depth
    t = []
    def blk(n, b, **last):
        t.append(_entry(n + '.dw', b.dw, **({'maxpool_after': last.pop('mp')} if 'mp' in last else {})))
        t.append(_entry(n + '.pw', b.pw, **last))
    blk('stem', net.stem); blk('stageA', net.stageA, route_target='dec_a.dw')
    blk('down1', net.down1, mp=(net.down1.downsample == 'maxpool')); blk('down2', net.down2, mp=(net.down2.downsample == 'maxpool'))
    for n in ('bott_c', 'bott_d1', 'bott_d2'):
        b = getattr(net, n)
        t.append(_entry(n + '.dw', b.dw))
        t.append(_entry(n + '.pw', b.pw, eltwise_add_with=n + '.block_input', eltwise_act=b.act_elt_name))
    blk('upconv_b', net.upconv_b, upsample_after=(ed >= 2)); blk('refine_b', net.refine_b)
    blk('upconv_a', net.upconv_a, upsample_after=(ed >= 1))
    t.append(_entry('dec_a.dw', net.dec_a.dw, route_source='upconv_a.pw+stageA.pw')); t.append(_entry('dec_a.pw', net.dec_a.pw))
    t.append(_entry('head_out', net.head_out))
    return t


def cost(topo, h, w, pf=8):
    """MACs, params, weight bytes, peak per-layer (in+out [+skip]) activation bytes with channels padded to pf."""
    macs = params = 0; peak = 0; hh, ww = h, w; skips = {}
    pad = lambda c: -(-c // pf) * pf
    for e in topo:
        k, ci, co = e['kernel_size'], e['in_ch'], e['out_ch']
        oh, ow = -(-hh // e['stride']), -(-ww // e['stride'])
        n = k * k * (ci if e['depthwise'] else ci * co)
        macs += n * oh * ow; params += n + 2 * co
        extra = pad(ci) * hh * ww if 'route_source' in e else 0   # concat second operand
        peak = max(peak, pad(ci) * hh * ww + pad(co) * oh * ow + extra)
        hh, ww = oh, ow
        if e.get('maxpool_after'): hh, ww = hh // 2, ww // 2
        if e.get('upsample_after'): hh, ww = hh * 2, ww * 2
    return dict(macs=macs, params=params, peak_act_bytes=peak)


def cutpoint(topo):
    srcs = []
    for i, e in enumerate(topo):
        if 'route_source' in e:
            late, early = e['route_source'].split('+')
            srcs += [next(j for j, x in enumerate(topo) if x['name'] == early)]
    return min(srcs)


def check(net, h, w, buf_size=0x8000, mem2_base=None):
    topo = topology(net)
    res = dict(cost(topo, h, w), cutpoint=cutpoint(topo), layers=len(topo), buf_size=buf_size)
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, 'net.txt'); open(inp, 'w').write(manifest_to_cfg({'topology': topo}))
        cmd = [sys.executable, os.path.join(GEN_CFG_DIR, 'gen_cfg.py'), '--input', inp, '--output', os.path.join(d, 'npu_cfg.c'),
               '--width', str(w), '--height', str(h), '--cutpoint', str(res['cutpoint']), '--parallelism', '8',
               '--buf-size', hex(buf_size)]
        p = subprocess.run(cmd, cwd=GEN_CFG_DIR, capture_output=True, text=True)
        out = p.stdout + p.stderr
    res['ok'] = p.returncode == 0
    m = re.search(r'Total param size: (\d+)', out); res['param_bytes'] = int(m.group(1)) if m else None
    for i in (1, 2, 3):
        m = re.search(rf'Buffer{i} requirement: (\d+)', out); res[f'buf{i}'] = int(m.group(1)) if m else None
    if not res['ok']:
        res['error'] = [l for l in out.strip().splitlines() if l.strip()][-1][:200]
    return res


def build(model, wm, n_frames=1, n_classes=4, stem_stride=1, encoder_depth=1):
    if model == 'npu_unet':
        return NPUUNet(n_classes, n_frames, width_mult=wm)
    return KuRALSNetNPUSeg(n_classes, n_frames, stem_stride=stem_stride, encoder_depth=encoder_depth, bottleneck_kernel_size=3,
                           dropout_rate=0.1, eltwise_act='linear', width_mult=wm)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['npu_unet', 'kuralsnet_npu_seg'], default='npu_unet')
    ap.add_argument('--width-mult', type=float, default=1.0)
    ap.add_argument('--height', type=int, default=8); ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--buf-size', type=lambda v: int(v, 0), default=0x8000)
    a = ap.parse_args()
    print(json.dumps(check(build(a.model, a.width_mult), a.height, a.width, a.buf_size), indent=1))


def npu_profile(net, h, w, limit=0x8000):
    """Peak NPU buffer demand (bytes) from gen_cfg.py's allocator run with an effectively unlimited buffer, so
    infeasible candidates still get a number. peak = max over the allocator's on-chip buffers; feasible = peak <= limit.
    See docs/NPU_BUFFER_MODEL.md for exactly what that number means."""
    r = check(net, h, w, buf_size=0x100000)
    peak = max([r[k] for k in ('buf1', 'buf2', 'buf3') if r.get(k) is not None] or [None]) if r['ok'] else None
    r.update(peak_npu_buffer=peak, feasible=bool(r['ok'] and peak <= limit), limit=limit,
             rejection=None if (r['ok'] and peak <= limit) else (r.get('error') or f'peak {peak} B > {limit} B'))
    return r


def generic_cost(net, h, w, n_frames=1, five_d=False):
    """Params, MACs (thop) and peak activation (largest in+out tensor of any leaf module, 1 byte/element = int8-equivalent)
    for a non-NPU model."""
    import torch
    from thop import profile
    net = net.eval()
    x = torch.randn(1, 1, n_frames, h, w) if five_d else torch.randn(1, n_frames, h, w)
    peak = [0]
    def hook(m, i, o):
        t = o[0] if isinstance(o, (tuple, list)) else o
        if torch.is_tensor(t) and not list(m.children()):
            peak[0] = max(peak[0], i[0].numel() + t.numel())
    hs = [m.register_forward_hook(hook) for m in net.modules()]
    with torch.no_grad():
        macs, params = profile(net, (x,), verbose=False)
    for hh in hs: hh.remove()
    return dict(macs=int(macs), params=int(sum(p.numel() for p in net.parameters())), peak_act_bytes=int(peak[0]))
