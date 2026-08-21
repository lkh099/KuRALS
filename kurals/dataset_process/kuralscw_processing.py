import os
import time
import numpy as np
import scipy.io as sio
import glob
import json
import argparse

# Native RD map is (124, 2048): axis 0 is Doppler (the "-2" offset applied to
# doppler_value below matches the [2:126, :] crop on this axis), axis 1 is Range.
NATIVE_DOPPLER_BINS = 124
NATIVE_RANGE_BINS = 2048


def _bin_groups(n_in, n_out):
    """Partition range(n_in) into n_out contiguous, near-equal groups (adaptive
    max-pool bucketing) -- used when n_in isn't evenly divisible by n_out."""
    return np.array_split(np.arange(n_in), n_out)


def _bin_lookup(n_in, n_out):
    """Map each of the n_in native indices to its output bin index."""
    lookup = np.empty(n_in, dtype=np.int64)
    for out_idx, group in enumerate(_bin_groups(n_in, n_out)):
        lookup[group] = out_idx
    return lookup


def resize_rd(data, doppler_bins, range_bins):
    """Downsample a native (124, 2048) RD map to (doppler_bins, range_bins) via
    block max-pooling. Max, not mean/bilinear: RD energy is sparse and peaky, so
    averaging would dilute a real target into its background neighbors. Used to
    match a deployment SoC's fixed RD buffer size."""
    pooled = np.stack([data[g].max(axis=0) for g in _bin_groups(data.shape[0], doppler_bins)])
    pooled = np.stack([pooled[:, g].max(axis=1) for g in _bin_groups(data.shape[1], range_bins)], axis=1)
    return pooled


def resize_rd_kth(data, doppler_bins, range_bins, k):
    """Like resize_rd, but each output cell takes the k-th largest value within its
    native block instead of the max (k=1 reduces to resize_rd). Targets a specific
    failure mode of max-pooling: the max of ~500 native pixels is an order statistic
    that grows with block size even for pure background/clutter (measured: median
    target-vs-background contrast collapses 19.4x -> 6.7x after max-pooling), because
    a single random elevated pixel is enough to win. A real target's native dense
    box is a *cluster* of ~9 correlated elevated pixels, not one spike, so its k-th
    largest value (for modest k) stays close to the true peak; a background block's
    k-th largest requires k coincidentally-elevated pixels, which is much less likely
    than needing just one -- suppressing the single-spike inflation without CFAR's
    dynamic-range compression (this stays in raw magnitude, no normalization).

    Not separable across axes the way max is (max-of-maxes equals the true max, but
    k-th-largest-of-row-k-ths does not equal the true 2D k-th-largest) -- operates on
    the full 2D block directly."""
    doppler_groups = _bin_groups(data.shape[0], doppler_bins)
    range_groups = _bin_groups(data.shape[1], range_bins)
    pooled = np.empty((doppler_bins, range_bins))
    for i, dg in enumerate(doppler_groups):
        for j, rg in enumerate(range_groups):
            block = data[np.ix_(dg, rg)].ravel()
            kk = min(k, block.size)
            pooled[i, j] = np.partition(block, -kk)[-kk]
    return pooled


def data_loader(data, seq_name, period, save_path, doppler_bins=None, range_bins=None):
    index_list = np.where(data['MTD_Scan']['tgtlD'][0][0][0] != 0)

    for index in index_list[0]:
        t_data = data['MTD_Scan']['data'][0][0][:, :, index]
        t_data = t_data[2:126, :]
        if doppler_bins is not None or range_bins is not None:
            t_data = resize_rd(t_data, doppler_bins or NATIVE_DOPPLER_BINS, range_bins or NATIVE_RANGE_BINS)
        print(str(period)+'_'+str(index))
        np.save(os.path.join(save_path, seq_name, 'range_doppler_numpy', str(period)+'_'+str(index)+'.npy'), t_data)

def data_process(data_path, save_path, doppler_bins=None, range_bins=None):
    print('***** Step 1/3: Generate data numpy file *****')
    time1 = time.time()
    with open(os.path.join(data_path, 'sequence.txt'), 'r', encoding='utf-8') as fp:
        seq_names = fp.readlines()
    seq_names = [seq.replace('\n', '') for seq in seq_names]
    print(seq_names)
    for seq_name in seq_names:
        print('*****' + seq_name + '*****')
        cw_data_list = glob.glob(os.path.join(data_path, seq_name, '*'))
        for cw_data in cw_data_list:
            period = cw_data.split('\\')[-1].split('.')[0]
            data = sio.loadmat(cw_data)
            data_loader(data, seq_name, period, save_path, doppler_bins, range_bins)

def get_sparse(data, label=None, points=None):
    if str(label) == '1':
        for point in points:
            data[1][point[0]][point[1]] = 1
            data[0][point[0]][point[1]] = 0
    elif str(label) == '2':
        for point in points:
            data[2][point[0]][point[1]] = 1
            data[0][point[0]][point[1]] = 0
    elif str(label) == '3':
        for point in points:
            data[3][point[0]][point[1]] = 1
            data[0][point[0]][point[1]] = 0
    return data

def get_box(rdCoord):
    x, y = rdCoord
    x = int(x)
    y = int(y)
    x_min = x - 1
    x_max = x + 1
    y_min = y - 1
    y_max = y + 1
    return [[x_min, y_min], [x_max, y_max]]

def get_dense(rdCoord):
    x, y = rdCoord
    x = int(x)
    y = int(y)
    return [[x-1, y-1], [x-1, y], [x-1, y+1],
            [x, y-1], [x, y], [x, y+1],
            [x+1, y-1], [x+1, y], [x+1, y+1]]

def get_onehot(data, label=None, points=None):
    if str(label) == '1':
        for point in points:
            data[1][point[0]][point[1]] = 1
            data[0][point[0]][point[1]] = 0
    elif str(label) == '2':
        for point in points:
            data[2][point[0]][point[1]] = 1
            data[0][point[0]][point[1]] = 0
    elif str(label) == '3':
        for point in points:
            data[3][point[0]][point[1]] = 1
            data[0][point[0]][point[1]] = 0
    return data

def get_mask(data, seq_name, frame, save_path, doppler_bins=None, range_bins=None, dilate_radius=0):
    annotations_path = os.path.join(save_path, seq_name, 'annotations')
    out_doppler = doppler_bins or NATIVE_DOPPLER_BINS
    out_range = range_bins or NATIVE_RANGE_BINS
    # Remap dense_points (built at native resolution below, so get_dense's 3x3
    # neighborhood stays meaningful) through the same bin groups as resize_rd,
    # so a target's mask cell always matches the RD map cell it was pooled into.
    doppler_lookup = _bin_lookup(NATIVE_DOPPLER_BINS, out_doppler) if doppler_bins else None
    range_lookup = _bin_lookup(NATIVE_RANGE_BINS, out_range) if range_bins else None

    index_list = np.where(data['MTD_Scan']['tgtlD'][0][0][0] != 0)
    for index in index_list[0]:
        dense_rd_data_background = np.ones([out_doppler, out_range])
        dense_rd_data = np.zeros([4, out_doppler, out_range])
        dense_rd_data[0] = dense_rd_data_background
        range_value = data['MTD_Scan']['tgtRange'][0][0][0][index]
        doppler_value = data['MTD_Scan']['tgtDoppler'][0][0][0][index] - 2

        point = [doppler_value, range_value]
        dense_points = get_dense(point)
        if doppler_lookup is not None or range_lookup is not None:
            remapped = set()
            for x, y in dense_points:
                x = doppler_lookup[np.clip(int(x), 0, NATIVE_DOPPLER_BINS - 1)] if doppler_lookup is not None else int(x)
                y = range_lookup[np.clip(int(y), 0, NATIVE_RANGE_BINS - 1)] if range_lookup is not None else int(y)
                # dilate_radius>0: mark a neighborhood around the remapped cell too, since
                # aggressive downsampling collapses the native 3x3 box to 1-2 output cells --
                # see resize_extracted_dataset.py's resize_mask docstring for the full rationale.
                for dx in range(-dilate_radius, dilate_radius + 1):
                    for dy in range(-dilate_radius, dilate_radius + 1):
                        xc, yc = x + dx, y + dy
                        if 0 <= xc < out_doppler and 0 <= yc < out_range:
                            remapped.add((xc, yc))
            dense_points = list(remapped)

        if seq_name == '2020年11月29日11时29分55秒_mtd_机场人车_true':
            type = data['MTD_Scan']['type'][0][0][0][index]
            if type == 0:  # 车
                dense_annotations = get_onehot(dense_rd_data, 3, dense_points)
            elif type == 1:  # 人
                dense_annotations = get_onehot(dense_rd_data, 2, dense_points)
        else:
            dense_annotations = get_onehot(dense_rd_data, 1, dense_points)
        print(str(frame) + '_' + str(index))
        save = os.path.join(annotations_path, 'dense', str(frame) + '_' + str(index))
        if not os.path.exists(save):
            os.makedirs(save)
        np.save(os.path.join(save, 'range_doppler.npy'), dense_annotations)

def mask_generate(data_path, save_path, doppler_bins=None, range_bins=None, dilate_radius=0):
    print('***** Step 2/3: Generate mask label *****')
    with open(os.path.join(data_path, 'sequence.txt'), 'r', encoding='utf-8') as fp:
        seq_names = fp.readlines()
    seq_names = [seq.replace('\n', '') for seq in seq_names]
    print(seq_names)

    for seq_name in seq_names:
        print('*****' + seq_name + '*****')
        cw_data_list = glob.glob(os.path.join(data_path, seq_name, '*'))
        for cw_data in cw_data_list:
            frame = cw_data.split('\\')[-1].split('.')[0]
            data = sio.loadmat(cw_data)
            get_mask(data, seq_name, frame, save_path, doppler_bins, range_bins, dilate_radius)

def annojson_genetate(data_path, save_path):
    print('***** Step 3/3: Generate annotation json file *****')
    json_save_path = os.path.join(save_path, 'annotations.json')
    with open(os.path.join(data_path, 'sequence.txt'), 'r', encoding='utf-8') as fp:
        seq_names = fp.readlines()
    seq_names = [seq.replace('\n', '') for seq in seq_names]
    print(seq_names)
    annotations = dict()

    for seq_name in seq_names:
        print('*****' + seq_name + '*****')
        annotations[str(seq_name)] = dict()
        cw_data_list = glob.glob(os.path.join(data_path, seq_name, '*'))
        for cw_data in cw_data_list:
            period = cw_data.split('\\')[-1].split('.')[0]
            annotations[str(seq_name)][str(period)] = dict()
            data = sio.loadmat(cw_data)
            # get_annotations(data, seq_name, frame, annotations)
            sparse_rd_data_background = np.ones([124, 2048])
            sparse_rd_data = np.zeros([4, 124, 2048])
            sparse_rd_data[0] = sparse_rd_data_background

            index_list = np.where(data['MTD_Scan']['tgtlD'][0][0][0] != 0)
            for index in index_list[0]:
                print(str(period) + '_' + str(index))
                annotations[str(seq_name)][str(period)][str(index)] = dict()
                range_value = data['MTD_Scan']['tgtRange'][0][0][0][index]
                doppler_value = data['MTD_Scan']['tgtDoppler'][0][0][0][index] - 2
                tgtID = data['MTD_Scan']['tgtlD'][0][0][0][index]
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)] = dict()
                tgtType = 'UAV'
                if seq_name == '2020年11月29日11时29分55秒_mtd_机场人车':
                    type = data['MTD_Scan']['type'][0][0][0][index]
                    if type == 0:
                        tgtType = 'vehicle'
                    elif type == 1:
                        tgtType = 'pedestrian'

                scPeriod = data['MTD_Scan']['scPeriod'][0][0][0][index]
                frmID = data['MTD_Scan']['frmID'][0][0][0][index]
                rdCoord = [int(doppler_value), int(range_value)]
                box = get_box(rdCoord)
                dense = get_dense(rdCoord)
                velocity = data['MTD_Scan']['tgtVelocity'][0][0][0][index]
                distance = data['MTD_Scan']['tgtDistance'][0][0][0][index]
                azimuth = data['MTD_Scan']['tgtAzimuth'][0][0][0][index]
                energy = data['MTD_Scan']['energy'][0][0][0][index]
                scr = data['MTD_Scan']['scr'][0][0][0][index]

                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['tgtType'] = tgtType
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['scPeriod'] = int(scPeriod)
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['frmID'] = int(frmID)
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['rdCoord'] = rdCoord
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['rdBox'] = box
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['rdDense'] = dense
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['velocity'] = float(velocity)
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['distance'] = float(distance)
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['azimuth'] = float(azimuth)
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['energy'] = float(energy)
                annotations[str(seq_name)][str(period)][str(index)][str(tgtID)]['scr'] = float(scr)
                
    with open(json_save_path, 'w') as fp:
        json.dump(annotations, fp, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='连续波雷达数据', dest="data_path", help='Path to complex-valued RD data.')
    parser.add_argument('--save-path', default='KuRALS_CW', dest="save_path", help='Path to save KuRALS-CW dataset.')
    parser.add_argument('--doppler-bins', type=int, default=None,
                        help='Downsample the Doppler axis (native 124) to this many bins via block max-pooling, '
                             'e.g. to match a deployment SoC RD buffer. Default: no resize.')
    parser.add_argument('--range-bins', type=int, default=None,
                        help='Downsample the Range axis (native 2048) to this many bins via block max-pooling. '
                             'Default: no resize.')
    parser.add_argument('--dilate-radius', type=int, default=0,
                        help='Mark a (2r+1)-wide neighborhood around each remapped target cell instead of just '
                             'the single cell -- denser positive labels for aggressively downsampled targets. '
                             'Default 0 (exact remap, no dilation).')
    args = parser.parse_args()

    data_path = args.data_path
    save_path = args.save_path

    data_process(data_path, save_path, args.doppler_bins, args.range_bins)
    mask_generate(data_path, save_path, args.doppler_bins, args.range_bins, args.dilate_radius)
    annojson_genetate(data_path, save_path)
    
