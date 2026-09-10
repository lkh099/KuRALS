"""Turn an export_int8.py manifest into a cfg_gen.py input file (Darknet-style
layer list), so the NPU config and the trained checkpoint can't drift apart.

cfg_gen.py does not derive bias_shift/act_shift itself -- it packs whatever
values the input file gives it into the register words (see the reference
tiny_yolov2 input, where they are hardcoded per layer). Those two values are
exactly export_int8()'s per-layer shift2/shift1, so they can only come from a
calibrated checkpoint; a hand-written topology file is not runnable without
them. Hence this generator.

'from' and 'layers' indices are raw 0-based positions in the emitted file (how
cfg_gen.py indexes cfg[], counting non-convolutional sections too), not
Darknet's usual negative-relative offsets.
"""
import argparse
import json


def manifest_to_cfg(manifest):
    lines, idx = [], 0
    positions = {}          # layer name -> index of its [convolutional] block

    def emit(text):
        nonlocal idx
        lines.append(text)
        idx += 1

    for layer in manifest['topology']:
        if 'route_source' in layer:
            # cfg_gen.py attaches a route to cfg[i+1], so it must sit immediately
            # before the conv that consumes it. Late operand first (still on-chip),
            # early one second (reloaded from DRAM).
            late, early = layer['route_source'].split('+')
            emit(f'[route]\nlayers={positions[late]},{positions[early]}')

        positions[layer['name']] = idx
        pad = (layer['kernel_size'] - 1) // 2
        block = [f"[convolutional]",
                 f"name={layer['name'].replace('.', '_')}",
                 f"size={layer['kernel_size']}",
                 f"stride={layer['stride']}",
                 f"filters={layer['out_ch']}",
                 f"pad={pad}"]
        if layer['depthwise']:
            block.append('depthwise=1')
        block += [f"activation={layer['act_name']}",
                  f"bias_shift={layer['shift2']}",
                  f"act_shift={layer['shift1']}"]
        emit('\n'.join(block))

        if layer.get('upsample_after'):
            emit('[upsample]\nstride=2')
        if 'eltwise_add_with' in layer:
            # Same constraint export_int8.py asserts (LEGAL_ELTWISE_ACTS) -- re-checked
            # here because a manifest can reach this script without going through that
            # path, and emitting `[shortcut] activation=leaky` would produce an
            # npu_cfg.c the hardware cannot actually run.
            if layer['eltwise_act'] != 'linear':
                raise ValueError(
                    f"{layer['name']}: [shortcut] activation={layer['eltwise_act']} is illegal -- "
                    "an eltwise-add must be followed by a conv before any activation. Only "
                    "activation=linear may sit on the add itself.")
            # This block is [dw, pw]; the tensor being added is whatever was emitted
            # two positions before the pw -- the preceding conv, or (for a chained
            # residual) the previous block's [shortcut], which cfg_gen.py resolves
            # back to its own preceding conv.
            block_input_idx = positions[layer['name']] - 2
            emit(f"[shortcut]\nfrom={block_input_idx}\nactivation={layer['eltwise_act']}")

    # cfg_gen.py reads cfg[idx+1] for every conv including the last one, so a
    # trailing section is required. [yolo] is the sentinel it recognizes as
    # "previous layer is the network output" -- not a claim this is a detector.
    emit('[yolo]')
    return '\n\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help="export_int8.py's <out>.json")
    parser.add_argument('--out', required=True)
    parser.add_argument('--header', default=None, help='Optional comment block prepended to the file.')
    args = parser.parse_args()

    with open(args.manifest, 'r') as fp:
        manifest = json.load(fp)
    body = manifest_to_cfg(manifest)
    if args.header:
        body = ''.join(f'# {line}\n' for line in args.header.splitlines()) + '\n' + body
    with open(args.out, 'w') as fp:
        fp.write(body)
    print(f"{args.manifest} -> {args.out} ({len(manifest['topology'])} conv layers, "
          f"final_upsample={manifest['final_upsample']})")


if __name__ == '__main__':
    main()
