import os
import time
import numpy as np
import scipy.io as sio
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pandas as pd
import torch
import argparse

from scipy.ndimage import maximum_filter
from kurals.models.cfar import CFAR2D_Parallel

DopShift = 2
# light_dataset_frame_oriented.json
orien_frame = dict()
# rd_stats_all.json
# stats = {'v_sum': 0, 'cnt_sum': 0, 'mean': 0.003965718, 'var': 0, 'min': 1, 'max': 0}
stats = {'v_sum': 0, 'cnt_sum': 0, 'mean': 0.0033517423680466895, 'std': 0, 'min': 1, 'max': 0}
# annotations.json
annotations = dict()
# rd_weights.json
rd_weights = {'pixel_sum': 0, 'pixel_uav': 0, 'pixel_car': 0, 'pixel_boat': 0, 'pixel_ped': 0}

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


def get_box(rdCoord, h, w):
    x, y = rdCoord
    x = int(x)
    y = int(y)
    x_min = max(x - 1, 0)
    x_max = min(x + 1, h-1)
    y_min = max(y - 2, 0)
    y_max = min(y + 2, w-1)
    return [[x_min, y_min], [x_max, y_max]]


def get_dense(rdCoord, h, w):
    x, y = rdCoord
    x = int(x)
    y = int(y)
    DenseMap = []
    pixel_cnt = 0
    for i in range(x-1, x+2):
        if i < 0 or i >= h:
            continue
        for j in range(y-2, y+3):
            if j >= 0 and j < w:
                DenseMap.append([i, j])
                pixel_cnt += 1
    return DenseMap, pixel_cnt

cfar = CFAR2D_Parallel(cfar_type='CA', alpha=1.2, guard_cells=1, ref_cells=1)

# Statistical analysis of CFAR false alarm rate, and the errors in CFAR and GPS results.
Num_FD = 0 # Number of false alarm targets
Num_Bg = 0 # Number of background units
Error_CFARandGPS_Range = [] # Range error units of CFAR and GPS
Error_CFARandGPS_Doppler = [] # Doppler error units for CFAR and GPS
Error_MaxandGPS_Range = [] # Range error units of local peak ​​and GPS
Error_MaxandGPS_Doppler = [] # Doppler error units for local peak ​​and GPS

def cfar_filter(AmpDataz, y_d, x_r, y_d_1, x_r_1):
    global Num_Bg
    global Num_FD
    # Peak detection using CFAR
    res_cfar = cfar.filter(torch.from_numpy(AmpDataz[None, None, :]))
    res_cfar = res_cfar.numpy() * AmpDataz
    res_cfar = res_cfar[0, 0]

    local_max = maximum_filter(res_cfar, size=3, mode='constant')
    nms = np.where(res_cfar == local_max, res_cfar, 0)

    nonzero_coords = np.argwhere(nms > 0)
    distances = np.linalg.norm(nonzero_coords - np.array([y_d, x_r]), axis=1)
    # print(distances)
    Num_FD = Num_FD + len(distances)
    # print(f'Num_FD: {Num_FD}')
    Num_Bg = Num_Bg + AmpDataz.shape[0] * AmpDataz.shape[1]
    # print(f'Num_Bg: {Num_Bg}')
    nearest_idx = np.argmin(distances)
    y_d_2, x_r_2 = nonzero_coords[nearest_idx]

    Error_CFARandGPS_Doppler.append(y_d_2 - y_d)
    Error_CFARandGPS_Range.append(x_r_2 - x_r)
    Error_MaxandGPS_Doppler.append(y_d_1 - y_d)
    Error_MaxandGPS_Range.append(x_r_1 - x_r)
    
    return y_d_2, x_r_2

def process_track(pulse_data, dest_path, seq_name, cnt_frame, maxz, maxk, idx_mat):
    for idx_frame in range(50):
        tgtID = pulse_data['MTD_Scan_all'][0, idx_frame]['tgtlD']
        idx_beam = np.argmax(tgtID)
        KuanZhaiFlag = pulse_data['MTD_Scan_all'][0, idx_frame]['tgtKZFlag'][0, idx_beam]
        if KuanZhaiFlag == -1:
            continue

        ## Acquiring narrow and wide band data
        dataz = pulse_data['MTD_Scan_all'][0, idx_frame]['dataz'][..., idx_beam]
        AmpDataz = np.abs(dataz)

        datak = pulse_data['MTD_Scan_all'][0, idx_frame]['datak'][..., idx_beam]
        AmpDatak = np.abs(datak)

        ## Remove absolute zero frequency
        # print(f'shape of AmpDataz is: {AmpDataz.shape}')
        hz, wz = AmpDataz.shape
        hk, wk = AmpDatak.shape

        ShiftAmpDataz = AmpDataz.copy()
        ShiftAmpDataz[0,:] = 0

        ShiftAmpDatak = AmpDatak.copy()
        ShiftAmpDatak[0, :] = 0

        if KuanZhaiFlag == 0:
            y_d = pulse_data['MTD_Scan_all'][0, idx_frame]['tgtDoppler'][0, idx_beam]
            x_r = pulse_data['MTD_Scan_all'][0, idx_frame]['tgtRange'][0, idx_beam]
            sizez = AmpDataz.shape
            local_data = AmpDataz[max(y_d-1, 0):min(y_d+2, sizez[0]), max(x_r-3, 0):min(x_r+4, sizez[1])]
            moffset_d, moffset_r = np.where(local_data == np.max(local_data))
            orioffset_d, orioffset_r = np.where(local_data == AmpDataz[y_d, x_r])
            y_d_1 = y_d + moffset_d[0] - orioffset_d[0]
            x_r_1 = x_r + moffset_r[0] - orioffset_r[0]

            y_d, x_r = cfar_filter(AmpDataz, y_d, x_r, y_d_1, x_r_1)

            z_amp = AmpDataz[y_d, x_r]
        elif KuanZhaiFlag == 1:
            y_d = pulse_data['MTD_Scan_all'][0, idx_frame]['tgtDoppler'][0, idx_beam]  # 起始值为0
            x_r = pulse_data['MTD_Scan_all'][0, idx_frame]['tgtRange'][0, idx_beam]
            sizek = AmpDatak.shape
            local_data = AmpDatak[max(y_d - 1, 0):min(y_d + 2, sizek[0]), max(x_r - 3, 0):min(x_r + 4, sizek[1])]
            moffset_d, moffset_r = np.where(local_data == np.max(local_data))
            orioffset_d, orioffset_r = np.where(local_data == AmpDatak[y_d, x_r])
            y_d_1 = y_d + moffset_d[0] - orioffset_d[0]
            x_r_1 = x_r + moffset_r[0] - orioffset_r[0]

            y_d, x_r = cfar_filter(AmpDatak, y_d, x_r, y_d_1, x_r_1)

            z_amp = AmpDatak[y_d, x_r]

        ## Normalization
        max_z = np.max(ShiftAmpDataz)
        max_k = np.max(ShiftAmpDatak)
        NormAmpDataz = ShiftAmpDataz / max_z
        NormAmpDatak = ShiftAmpDatak / max_k
        if KuanZhaiFlag == 0:
            NormZamp = z_amp / max_z
        elif KuanZhaiFlag == 1:
            NormZamp = z_amp / max_k

        ## Wide and narrow band data combination
        JointData = np.zeros(shape=(64, 800))
        if 'Air' in seq_name:
            JointData[:, 18:174] = NormAmpDataz[:, 0:156]
            JointData[:, 174:218] = (NormAmpDataz[:, 156:200] + NormAmpDatak[:, 0:44]) / 2
            JointData[:, 218:794] = NormAmpDatak[:, 44:620]

            if KuanZhaiFlag == 0:
                x_r = x_r + 18
            else:
                x_r = x_r + 174

        elif 'Land' in seq_name:
            JointData[:, 0:114] = NormAmpDataz[:, 0:114]
            JointData[:, 114:154] = (NormAmpDataz[:, 114:154] + NormAmpDatak[:, 0:40]) / 2
            JointData[:, 154:594] = NormAmpDatak[:, 40:480]

            if KuanZhaiFlag == 1:
                x_r = x_r + 114

        JointData = np.concatenate([JointData[32:64, :], JointData[0:32, :]], axis=0)
        if y_d <= 31:
            yj_d = y_d + 32
        else:
            yj_d = y_d - 32

        # print(np.max(JointData))
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # xx = np.arange(0, 64, 1)
        # yy = np.arange(0, 800, 1)
        # xx_mesh, yy_mesh = np.meshgrid(xx, yy, indexing='ij')
        # fig = plt.figure()
        # sub = plt.axes(projection='3d')
        # # sub = fig.add_subplot(111, projection='3d')
        # sub.plot_surface(xx_mesh, yy_mesh, JointData, rstride=1, cstride=1, cmap='Purples')
        # print(np.max(JointData))
        # print(np.where(JointData == np.max(JointData)))
        #
        # sub.set_xlabel('多普勒单元')
        # sub.set_ylabel('距离单元')
        # sub.set_zlabel('幅值')
        # sub.set_zlim(0, 1)
        # sub.set_title(f'窄宽脉冲拼接数据，Mat index {idx_mat}, Frame index {idx_frame}, Saved Frame Index {cnt_frame}')
        # #
        # sub.scatter(xs=yj_d, ys=x_r, zs=NormZamp, s=30, c='red', marker='^')
        # print(yj_d, x_r, NormZamp)
        # plt.show()

        ## 保存数据
        # light_dataset_frame_oriented.json
        orien_frame[seq_name].append(f'%06d'%cnt_frame)

        stats['v_sum'] = stats['v_sum'] + np.sum((JointData-stats['mean'])**2)
        # stats['v_sum'] = stats['v_sum'] + np.sum(JointData)
        stats['cnt_sum'] = stats['cnt_sum'] + 64*800
        v_min = np.min(JointData)
        v_max = np.max(JointData)
        if v_min < stats['min']:
            stats['min'] = v_min
        if v_max > stats['max']:
            stats['max'] = v_max

        # annotations.json
        annotations[seq_name][f'%06d'%cnt_frame] = dict()
        if 'UAV' in seq_name:
            tgtType = 'UAV'
        elif 'Ped' in seq_name:
            tgtType = 'Pedestrian'
        elif 'Car' in seq_name:
            tgtType = 'Car'
        else:
            raise KeyError(f'for the track mode, the target type in sequence {seq_name} is not valid.')

        annotations[seq_name][f'%06d'%cnt_frame]['tgtType'] = tgtType
        annotations[seq_name][f'%06d'%cnt_frame]['tgtPulse'] = 'Narrow' if KuanZhaiFlag == 0 else 'Broad'
        annotations[seq_name][f'%06d'%cnt_frame]['scPeriod'] = int(pulse_data['MTD_Scan_all'][0, idx_frame]['scPeriod'][0, idx_beam])
        annotations[seq_name][f'%06d'%cnt_frame]['frmID'] = int(pulse_data['MTD_Scan_all'][0, idx_frame]['frmID'][0, idx_beam])
        annotations[seq_name][f'%06d'%cnt_frame]['Distance'] = float(pulse_data['MTD_Scan_all'][0, idx_frame]['tgtDistance'][0, idx_beam])
        annotations[seq_name][f'%06d'%cnt_frame]['Velocity'] = float(pulse_data['MTD_Scan_all'][0, idx_frame]['tgtVelocity'][0, idx_beam])
        annotations[seq_name][f'%06d'%cnt_frame]['Azimuth'] = float(pulse_data['MTD_Scan_all'][0, idx_frame]['tgtAzimuth'][0, idx_beam])
        annotations[seq_name][f'%06d'%cnt_frame]['Height'] = float(pulse_data['MTD_Scan_all'][0, idx_frame]['tgtHeight'][0, idx_beam])
        annotations[seq_name][f'%06d'%cnt_frame]['EleBeamIndex'] = int(pulse_data['MTD_Scan_all'][0, idx_frame]['tgtEleBeamIndex'][0, idx_beam])
        annotations[seq_name][f'%06d'%cnt_frame]['energy'] = float(pulse_data['MTD_Scan_all'][0, idx_frame]['energy'][0, idx_beam])
        annotations[seq_name][f'%06d'%cnt_frame]['scr'] = float(pulse_data['MTD_Scan_all'][0, idx_frame]['scr'][0, idx_beam])

        # label
        rdCoord = [int(yj_d), int(x_r)]
        rdBox = get_box(rdCoord, 64, 800)
        rdDense, pixel_cnt = get_dense(rdCoord, 64, 800)

        annotations[seq_name][f'%06d'%cnt_frame]['rdCoord'] = rdCoord
        annotations[seq_name][f'%06d'%cnt_frame]['rdBox'] = rdBox
        annotations[seq_name][f'%06d'%cnt_frame]['rdDense'] = rdDense

        # Store data
        np.save(os.path.join(dest_path, seq_name, 'range_doppler_numpy', f'%06d.npy'%cnt_frame), JointData)

        # Get category weight
        rd_weights['pixel_sum'] += 64*800
        if 'UAV' in seq_name:
            rd_weights['pixel_uav'] += pixel_cnt
        elif 'Ped' in seq_name:
            rd_weights['pixel_ped'] += pixel_cnt
        elif 'Car' in seq_name:
            rd_weights['pixel_car'] += pixel_cnt
        else:
            raise KeyError(f'for the track mode, the target type in sequence {seq_name} is not valid.')

        cnt_frame = cnt_frame + 1

    return cnt_frame


def process_scan(pulse_data, dest_path, seq_name, cnt_frame, maxz, maxk, idx_mat):
    tgtID = pulse_data['MTD_Scan_all'][0,0]['tgtlD'] # shape of (1, 1130)
    idx_beam = np.argmax(tgtID)
    # print('idx_beam: ', idx_beam)
    KuanZhaiFlag = pulse_data['MTD_Scan_all'][0,0]['tgtKZFlag'][0, idx_beam]
    # KuanZhaiFlag = pulse_data['MTD_Scan_all'][0, 0]['tgtKZFlag']
    # print('KuanZhaiFlag: ', KuanZhaiFlag)
    # import pdb; pdb.set_trace()

    if KuanZhaiFlag == -1:
        return cnt_frame

    dataz = pulse_data['MTD_Scan_all'][0,0]['dataz'][..., idx_beam]
    AmpDataz = np.abs(dataz)

    datak = pulse_data['MTD_Scan_all'][0,0]['datak'][..., idx_beam]
    AmpDatak = np.abs(datak)

    # print(f'shape of AmpDataz is: {AmpDataz.shape}')
    hz, wz = AmpDataz.shape
    hk, wk = AmpDatak.shape

    ShiftAmpDataz = AmpDataz.copy()
    ShiftAmpDataz[0,:] = 0

    ShiftAmpDatak = AmpDatak.copy()
    ShiftAmpDatak[0, :] = 0

    if KuanZhaiFlag == 0:
        y_d = pulse_data['MTD_Scan_all'][0,0]['tgtDoppler'][0,idx_beam]
        x_r = pulse_data['MTD_Scan_all'][0,0]['tgtRange'][0,idx_beam]
        sizez = AmpDataz.shape
        
        local_data = AmpDataz[max(y_d - 1, 0):min(y_d + 2, sizez[0]), max(x_r - 3, 0):min(x_r + 4, sizez[1])]
        moffset_d, moffset_r = np.where(local_data == np.max(local_data))
        orioffset_d, orioffset_r = np.where(local_data == AmpDataz[y_d, x_r])
        y_d_1 = y_d + moffset_d[0] - orioffset_d[0]
        x_r_1 = x_r + moffset_r[0] - orioffset_r[0]

        y_d, x_r = cfar_filter(AmpDataz, y_d, x_r, y_d_1, x_r_1)

        z_amp = AmpDataz[y_d, x_r]
    elif KuanZhaiFlag == 1:
        y_d = pulse_data['MTD_Scan_all'][0,0]['tgtDoppler'][0,idx_beam]
        x_r = pulse_data['MTD_Scan_all'][0,0]['tgtRange'][0,idx_beam]
        sizek = AmpDatak.shape

        local_data = AmpDatak[max(y_d - 1, 0):min(y_d + 2, sizek[0]), max(x_r - 3, 0):min(x_r + 4, sizek[1])]
        moffset_d, moffset_r = np.where(local_data == np.max(local_data))
        orioffset_d, orioffset_r = np.where(local_data == AmpDatak[y_d, x_r])
        y_d_1 = y_d + moffset_d[0] - orioffset_d[0]
        x_r_1 = x_r + moffset_r[0] - orioffset_r[0]

        y_d, x_r = cfar_filter(AmpDatak, y_d, x_r, y_d_1, x_r_1)

        z_amp = AmpDatak[y_d, x_r]

    ## Normalization
    max_z = np.max(ShiftAmpDataz)
    max_k = np.max(ShiftAmpDatak)
    NormAmpDataz = ShiftAmpDataz / max_z
    NormAmpDatak = ShiftAmpDatak / max_k
    if KuanZhaiFlag == 0:
        NormZamp = z_amp / max_z
    elif KuanZhaiFlag == 1:
        NormZamp = z_amp / max_k

    ## Wide and narrow band data combination
    JointData = np.zeros(shape=(64, 800))
    JointData[:, 5:119] = NormAmpDataz[:, 0:114]
    JointData[:, 119:137] = (NormAmpDataz[:, 114:132] + NormAmpDatak[:, 0:18]) / 2
    JointData[:, 137:679] = NormAmpDatak[:, 18:560]

    ## Place zero frequency in the middle position
    JointData = np.concatenate([JointData[32:64, :], JointData[0:32, :]], axis=0)
    if y_d <= 31:
        yj_d = y_d + 32
    else:
        yj_d = y_d - 32

    if KuanZhaiFlag == 0:
        x_r = x_r + 5
    else:
        x_r = x_r + 119

    # print(np.max(JointData))

    ## 数据可视化
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # xx = range(64)
    # yy = range(800)
    # xx_mesh, yy_mesh = np.meshgrid(xx, yy, indexing='ij')
    # fig = plt.figure(0)
    # sub = fig.add_subplot(111, projection='3d')
    # sub.plot_surface(xx_mesh, yy_mesh, JointData, cmap='jet')
    #
    # sub.set_xlabel('多普勒单元')
    # sub.set_ylabel('距离单元')
    # sub.set_zlabel('幅值')
    # sub.set_zlim(0, 1)
    # sub.set_title(f'窄宽脉冲拼接数据，Mat index {idx_mat}, Saved Frame Index {cnt_frame}')
    #
    # sub.scatter(xs=yj_d, ys=x_r, zs=NormZamp, s=30, c='cyan', marker='^')
    # plt.show()

    # 保存数据
    # light_dataset_frame_oriented.json
    orien_frame[seq_name].append(f'%06d'%cnt_frame)

    # rd_stats_all.json
    stats['v_sum'] = stats['v_sum'] + np.sum((JointData-stats['mean'])**2)
    # stats['v_sum'] = stats['v_sum'] + np.sum(JointData)
    stats['cnt_sum'] = stats['cnt_sum'] + 64 * 800
    v_min = np.min(JointData)
    v_max = np.max(JointData)
    if v_min < stats['min']:
        stats['min'] = v_min
    if v_max > stats['max']:
        stats['max'] = v_max

    # annotations.json
    annotations[seq_name][f'%06d'%cnt_frame] = dict()
    if 'UAV' in seq_name:
        tgtType = 'UAV'
    elif 'Boat' in seq_name:
        tgtType = 'Boat'
    else:
        raise KeyError(f'for the scan mode, the target type in sequence {seq_name} is not valid.')

    annotations[seq_name][f'%06d'%cnt_frame]['tgtType'] = tgtType
    annotations[seq_name][f'%06d'%cnt_frame]['tgtPulse'] = 'Narrow' if KuanZhaiFlag == 0 else 'Broad'
    annotations[seq_name][f'%06d'%cnt_frame]['scPeriod'] = \
    int(pulse_data['MTD_Scan_all'][0,0]['scPeriod'][0, idx_beam])

    annotations[seq_name][f'%06d'%cnt_frame]['frmID'] = int(pulse_data['MTD_Scan_all'][0,0]['frmID'][0, idx_beam])
    annotations[seq_name][f'%06d'%cnt_frame]['Distance'] = \
    float(pulse_data['MTD_Scan_all'][0,0]['tgtDistance'][0, idx_beam])

    annotations[seq_name][f'%06d'%cnt_frame]['Velocity'] = \
    float(pulse_data['MTD_Scan_all'][0,0]['tgtVelocity'][0, idx_beam])

    annotations[seq_name][f'%06d'%cnt_frame]['Azimuth'] = \
    float(pulse_data['MTD_Scan_all'][0,0]['tgtAzimuth'][0, idx_beam])

    annotations[seq_name][f'%06d'%cnt_frame]['Height'] = \
    float(pulse_data['MTD_Scan_all'][0,0]['tgtHeight'][0, idx_beam])

    annotations[seq_name][f'%06d'%cnt_frame]['EleBeamIndex'] = \
    int(pulse_data['MTD_Scan_all'][0,0]['tgtEleBeamIndex'][0, idx_beam])

    annotations[seq_name][f'%06d'%cnt_frame]['energy'] = float(pulse_data['MTD_Scan_all'][0,0]['energy'][0, idx_beam])
    annotations[seq_name][f'%06d'%cnt_frame]['scr'] = float(pulse_data['MTD_Scan_all'][0,0]['scr'][0, idx_beam])

    # label
    rdCoord = [int(yj_d), int(x_r)]
    rdBox = get_box(rdCoord, 64, 800)
    rdDense, pixel_cnt = get_dense(rdCoord, 64, 800)

    annotations[seq_name][f'%06d'%cnt_frame]['rdCoord'] = rdCoord
    annotations[seq_name][f'%06d'%cnt_frame]['rdBox'] = rdBox
    annotations[seq_name][f'%06d'%cnt_frame]['rdDense'] = rdDense

    # save data
    np.save(os.path.join(dest_path, seq_name, 'range_doppler_numpy', f'%06d.npy'%cnt_frame), JointData)

    # Get category weight
    rd_weights['pixel_sum'] += 64 * 800
    if 'UAV' in seq_name:
        rd_weights['pixel_uav'] += pixel_cnt
    elif 'Boat' in seq_name:
        rd_weights['pixel_boat'] += pixel_cnt
    else:
        raise KeyError(f'for the scan mode, the target type in sequence {seq_name} is not valid.')

    cnt_frame = cnt_frame + 1

    return cnt_frame


def pthMap(seq_name, data_path):
    if seq_name == '20191210_145742_Ku5km_SeaBoat':
        src_dir = data_path + '/Ku_5Km对海雷达/数据1_CH1_2019_12_10_15_08_37'
        paths_data = [os.path.join(src_dir, pth) for pth in os.listdir(src_dir) if 'Boat' in pth]
        maxk = 1311866
        maxz = 1192393
    elif seq_name == '20191210_145742_Ku5km_SeaUAV':
        src_dir = data_path + '/Ku_5Km对海雷达/数据1_CH1_2019_12_10_15_08_37'
        paths_data = [os.path.join(src_dir, pth) for pth in os.listdir(src_dir) if 'UAV' in pth]
        maxk = 1961090
        maxz = 94205
    elif seq_name == '20191210_192508_Ku5km_SeaBoat':
        src_dir = data_path + '/Ku_5Km对海雷达/数据2_CH1_2019_12_10_19_31_25'
        paths_data = [os.path.join(src_dir, pth) for pth in os.listdir(src_dir) if 'Boat' in pth]
        maxk = 1197958
        maxz = 189875
    elif seq_name == '20191210_192508_Ku5km_SeaUAV':
        src_dir = data_path + '/Ku_5Km对海雷达/数据2_CH1_2019_12_10_19_31_25'
        paths_data = [os.path.join(src_dir, pth) for pth in os.listdir(src_dir) if 'UAV' in pth]
        maxk = 404583
        maxz = 77058
    elif seq_name == '20211110_155325_Ku5km_AirUAV':
        src_dir = data_path + '/Ku_5Km雷达'
        paths_data = [os.path.join(src_dir, pth) for pth in os.listdir(src_dir) if '.mat' in pth]
        maxk = 664198
        maxz = 1167814
    elif seq_name == '20220413_141133_Ku3km_LandCar':
        src_dir = data_path + '/Ku_3km对地雷达/车辆数据1'
        paths_data = [os.path.join(src_dir, pth) for pth in os.listdir(src_dir) if '.mat' in pth]
        maxk = 41382431
        maxz = 7958526
    elif seq_name == '20220413_144137_Ku3km_LandCar':
        src_dir = data_path + '/Ku_3km对地雷达/车辆数据2'
        paths_data = [os.path.join(src_dir, pth) for pth in os.listdir(src_dir) if '.mat' in pth]
        maxk = 24599212
        maxz = 7191322
    elif seq_name == '20220413_150158_Ku3km_LandPed':
        src_dir = data_path + '/Ku_3km对地雷达/行人数据1'
        paths_data = [os.path.join(src_dir, pth) for pth in os.listdir(src_dir) if '.mat' in pth]
        maxk = 7930399
        maxz = 8129926
    elif seq_name == '20220413_151837_Ku3km_LandPed':
        src_dir = data_path + '/Ku_3km对地雷达/行人数据2'
        paths_data = [os.path.join(src_dir, pth) for pth in os.listdir(src_dir) if '.mat' in pth]
        maxk = 4243593
        maxz = 7774769
    else:
        raise FileNotFoundError(f'the name of sequence: {seq_name} is not valid')

    return paths_data, maxk, maxz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='脉冲雷达数据', dest="data_path", help='Path to complex-valued RD data.')
    parser.add_argument('--save-path', default='KuRALS_PD', dest="save_path", help='Path to save KuRALS-PD dataset.')
    args = parser.parse_args()
    
    data_path = args.data_path
    save_path = args.save_path
    
    print('***** process mat file to numpy file and generate annotation *****')
    start_time = time.time()

    # sequence.txt
    with open(os.path.join(save_path, 'sequence.txt'), 'r', encoding='utf-8') as fp:
        seq_names = fp.readlines()
    seq_names = [seq.replace('\n', '') for seq in seq_names]
    # print(seq_names)
    for seq_name in seq_names:
        print('***** processing sequence: ' + seq_name + '*****')
        paths_data, maxk, maxz = pthMap(seq_name, data_path)

        if not os.path.exists(os.path.join(save_path, seq_name, 'range_doppler_numpy')):
            os.makedirs(os.path.join(save_path, seq_name, 'range_doppler_numpy'))

        # light_dataset_frame_oriented.json
        orien_frame[seq_name] = []
        # annotations.json
        annotations[seq_name] = dict()

        # Total number of frames in the current sequence
        cnt_frame = 0
        for idx_mat, path_data in enumerate(tqdm(paths_data, desc='processing: ')):
            # print(path_data)
            pulse_data = sio.loadmat(path_data)

            if 'Sea' in seq_name:
                cnt_frame = process_scan(pulse_data=pulse_data, dest_path=save_path, seq_name=seq_name, cnt_frame=cnt_frame, maxz=maxz, maxk=maxk, idx_mat=idx_mat)
            else:
                cnt_frame = process_track(pulse_data=pulse_data, dest_path=save_path, seq_name=seq_name, cnt_frame=cnt_frame, maxz=maxz, maxk=maxk, idx_mat=idx_mat)
        print(f'the total number of frames is {cnt_frame}')

    def Error_Distribute(Error_Array, Type_Error):
        print('shape of Error_Array: {}'.format(len(Error_Array)))
        Error_Array = np.array(Error_Array)
        error_median = np.median(Error_Array)
        error_IQR = np.percentile(Error_Array, 75) - np.percentile(Error_Array, 25)
        error_95th = np.percentile(Error_Array, 95)
        erros_exceed_1bin = np.mean(np.abs(Error_Array) > 1)
        print(f'{Type_Error}: median {error_median}, IQR {error_IQR}, 95th {error_95th}')

    FAR = Num_FD / Num_Bg
    print(f'False Alarm Rate: {FAR}')
    Error_Distribute(Error_CFARandGPS_Range, 'Error_CFARandGPS_Range')
    Error_Distribute(Error_CFARandGPS_Doppler, 'Error_CFARandGPS_Doppler')
    Error_Distribute(Error_MaxandGPS_Range, 'Error_MaxandGPS_Range')
    Error_Distribute(Error_MaxandGPS_Doppler, 'Error_MaxandGPS_Doppler')

    with open(os.path.join(save_path, 'light_dataset_frame_oriented.json'), 'w') as f:
        f.write(json.dumps(orien_frame))
    
    # stats['mean'] = (stats['v_sum'] + 0.) / stats['cnt_sum']
    
    with open(os.path.join(save_path, 'annotations.json'), 'w') as f:
        json.dump(annotations, f, ensure_ascii=False)
    
    with open(os.path.join(save_path, 'rd_weights.json'), 'w') as f:
        f.write(json.dumps(rd_weights))
    
    print(((stats['v_sum'] + 0.) / stats['cnt_sum']))
    stats['std'] = np.sqrt((stats['v_sum'] + 0.) / stats['cnt_sum'])
    with open(os.path.join(save_path, 'rd_stats_all.json'), 'w') as f:
        f.write(json.dumps(stats))
