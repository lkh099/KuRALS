import os
import time
import numpy as np
import scipy.io as sio
import glob
import json
import argparse

def data_loader(data, seq_name, period, save_path):
    index_list = np.where(data['MTD_Scan']['tgtlD'][0][0][0] != 0)

    for index in index_list[0]:
        t_data = data['MTD_Scan']['data'][0][0][:, :, index]
        t_data = t_data[2:126, :]
        print(str(period)+'_'+str(index))
        np.save(os.path.join(save_path, seq_name, 'range_doppler_numpy', str(period)+'_'+str(index)+'.npy'), t_data)

def data_process(data_path, save_path):
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
            data_loader(data, seq_name, period, save_path)

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

def get_mask(data, seq_name, frame, save_path):
    annotations_path = os.path.join(save_path, seq_name, 'annotations')

    index_list = np.where(data['MTD_Scan']['tgtlD'][0][0][0] != 0)
    for index in index_list[0]:
        dense_rd_data_background = np.ones([124, 2048])
        dense_rd_data = np.zeros([4, 124, 2048])
        dense_rd_data[0] = dense_rd_data_background
        range_value = data['MTD_Scan']['tgtRange'][0][0][0][index]
        doppler_value = data['MTD_Scan']['tgtDoppler'][0][0][0][index] - 2

        point = [doppler_value, range_value]
        dense_points = get_dense(point)

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

def mask_generate(data_path, save_path):
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
            get_mask(data, seq_name, frame, save_path)

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
    args = parser.parse_args()
    
    data_path = args.data_path
    save_path = args.save_path
    
    data_process(data_path, save_path)
    mask_generate(data_path, save_path)
    annojson_genetate(data_path, save_path)
    
