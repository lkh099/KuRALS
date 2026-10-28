"""Collect best-val-checkpoint metrics (val_doppler_results.json also holds the test metrics at that same checkpoint)."""
import glob, json, os, csv, sys
rows = []
for d in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', '*'))):
    f = glob.glob(d + '/logs/KuRALS_CW/*/*/results/val_doppler_results.json')
    if not f: continue
    r = json.load(open(f[0])); v = r['val_metrics']['range_doppler']; t = r['test_metrics']['range_doppler']
    rows.append(dict(run=os.path.basename(d), best_epoch=r['epoch'], val_miou=round(v['miou'], 4), val_dice=round(v['dice'], 4),
                     test_miou=round(t['miou'], 4), test_dice=round(t['dice'], 4), test_uav_iou=round(t['miou_by_class'][1], 4)))
w = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys()); w.writeheader(); [w.writerow(x) for x in rows]
