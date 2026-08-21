import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
import copy
import os
import torch
import torch.nn as nn

# Author: hzj.2024.07.29
# Content: 1维2维检测器
# 三种门限选择：
    # CA，参考单元均值作为门限
    # SO，参考单元最小值为门限，isAll为True时选参考单元所有点的最小值，否则为左右两边均值中最小的为参考
    # GO，参考单元最大值为门限，isAll为True时选参考单元所有点的最大值，否则为左右两边均值中最大的为参考
    # 取所有点中最小（大）为常规逻辑，而取两边均值最小的策略参考： https://blog.csdn.net/weixin_45317919/article/details/125899133
    # 二维CFAR没有左右的问题，就直接选最小（大）为门限
# 门限系数通过get_factor函数确定，第一次计算时较慢，后续会直接读取第一次计算的结果

# Author: liteng.2024.08.03，参考自hzj代码
# Content: 并行加速的2维检测器，见class CFAR2D_Parallel
# 四种门限选择：
    # CA，参考单元均值作为门限
    # SO，参考单元最小值为门限，参考单元最小值取上下左右四个方向均值中最小值
    # GO，参考单元最大值为门限，参考单元最大值取上下左右四个方向均值中最大值
    # OS，参考单元中值为门限
# 门限系数依赖手动设置

class Detector():
    winset = None
    factor = None
    paraPath = './save/detectorPara'
    pfa = 1e-3

    def __init__(self,protectLen = 2,referenceLen = 3,winst = None,factor = None,pfa = 1e-3):
        if not os.path.exists(self.paraPath):
            os.makedirs(self.paraPath)
        self.pfa = pfa
        if winst is None:
            self.winset = self.consWin(protectLen,referenceLen)
        else:
            self.winset = winst
        
        if factor is None:
            self.get_factor()
        else:
            self.factor = factor
        

    @abstractmethod
    def consWin(self,protectLen,referenceLen):
        pass
    
    @abstractmethod
    def get_threshold(self,data):
        pass

    @abstractmethod
    def get_factor(self):
        pass

    def filter(self,data):
        # data = torch.Tensor(abs(data))
        data = abs(data)
        if self.factor is None:
            threshold = self.get_threshold(data)
        else:
            threshold = self.factor*self.get_threshold(data)
        result = copy.deepcopy(data)
        result[abs(result)<abs(threshold)] = 0
        return result,threshold

class CFAR_1D(Detector):
    CFARType = 'CA'
    ProtectLen = 0
    ReferenceLen = 3

    def __init__(self,CFARType = 'CA', protectLen=2, referenceLen=3, winst=None,factor = None,pfa = 1e-3):
        super().__init__(protectLen, referenceLen, winst,factor,pfa)
        self.CFARType = CFARType
        self.ProtectLen = protectLen
        self.ReferenceLen = referenceLen

    # def consWin(self,protectLen=2, referenceLen=3):
    #     refUnits = np.ones(referenceLen)
    #     proUnits = np.zeros(protectLen)
    #     win = np.concatenate((refUnits,proUnits,np.zeros(1),proUnits,refUnits))
    #     return win.squeeze()
        
    def consWin(self, protectLen = 2, referenceLen = 3):
        win = torch.ones(1+2*protectLen+2*referenceLen)
        win[referenceLen:referenceLen+2*protectLen+1] = 0
        return win
    
    def get_factor(self):
        
        path = self.paraPath+'/'+f'cfar1d_{self.CFARType}_{self.pfa}.npy'
        if os.path.exists(path):
            self.factor = np.load(path).item()
        else:
            
            num = int(1e2/self.pfa)
            noise = (np.random.randn(num)+1j*np.random.randn(num))*0.5**0.5
            _,threshold = self.filter(abs(noise))
            oc = abs(noise)/threshold.real
            oc.sort()
            self.factor = oc[-100]
            np.save(path,self.factor)
        

    def get_threshold(self, data):
        return {'CA':self.get_CA_threshold,
                'SO':self.get_SO_threshold,
                'GO':self.get_GO_threshold
                }[self.CFARType](data)

    
    def get_CA_threshold(self,data):
        # data = data.squeeze()
        # win = self.winset[::-1] #反转
        win = self.winset.flip(0) #反转
        N = win.shape[0]+data.shape[0]-1
        result = torch.fft.ifft(torch.fft.fft(data,n=N)*torch.fft.fft(win,n=N))
        cout = torch.fft.ifft(torch.fft.fft(torch.ones_like(data),n=N)*torch.fft.fft(win,n=N))
        result /= cout
        return result[int(win.shape[0]/2):int(win.shape[0]/2)+data.shape[0]]
    
    def get_SO_threshold(self,data,isAll = False):
        threshold = torch.zeros_like(data).to(torch.float)
        L = len(data)
        for i in range(L):
            l = range(i-self.ProtectLen-self.ReferenceLen,i-self.ProtectLen)
            if (i+self.ProtectLen)<L and (i+self.ProtectLen+self.ReferenceLen+1)>=L:
                r = np.hstack((range(i+self.ProtectLen+1,L),range(0,(i+self.ProtectLen+self.ReferenceLen+1)%L))).astype(int)
            else:
                r = range((i+self.ProtectLen+1)%L,(i+self.ProtectLen+self.ReferenceLen+1)%L)
            if isAll:
                min_l = abs(data[l]).min()
                min_r = abs(data[r]).min()
            else:
                min_l = abs(data[l]).mean()
                min_r = abs(data[r]).mean()
            threshold[i] = min(min_l,min_r)
        return threshold
    
    def get_GO_threshold(self,data,isAll = False):
        threshold = torch.zeros_like(data).to(torch.float)
        L = len(data)
        for i in range(L):
            l = range(i-self.ProtectLen-self.ReferenceLen,i-self.ProtectLen)
            if (i+self.ProtectLen)<L and (i+self.ProtectLen+self.ReferenceLen)>=L:
                r = np.hstack((range(i+self.ProtectLen,L),range(0,(i+self.ProtectLen+self.ReferenceLen)%L))).astype(int)
            else:
                r = range((i+self.ProtectLen+1)%L,(i+self.ProtectLen+self.ReferenceLen+1)%L)
            if isAll:
                max_l = abs(data[l]).max()
                max_r = abs(data[r]).max()
            else:
                max_l = abs(data[l]).mean()
                max_r = abs(data[r]).mean()
            threshold[i] = max(max_l,max_r)
        return threshold
    

class CFAR_2D(Detector):
    ProtectLen = 0
    ReferenceLen = 3

    def __init__(self,CFARType = 'CA', protectLen=2, referenceLen=3, winst=None,factor = None,pfa= 1e-3):
        super().__init__( protectLen, referenceLen, winst,factor,pfa)
        self.CFARType = CFARType
        self.ProtectLen = protectLen
        self.ReferenceLen = referenceLen

    def consWin(self,protectLen=2, referenceLen=3):
        win = torch.ones((1+2*protectLen+2*referenceLen,1+2*protectLen+2*referenceLen))
        win[referenceLen:referenceLen+2*protectLen+1,referenceLen:referenceLen+2*protectLen+1] = 0
        return win
    
    def get_factor(self):
        path = self.paraPath+'/'+f'cfar2d_{self.CFARType}_{self.pfa}.npy'
        if os.path.exists(path):
            self.factor = np.load(path).item()
        else:
            num = int((1e2/self.pfa)**0.8)
            noise = (np.random.randn(num,num)+1j*np.random.randn(num,num))*0.5**0.5
            _,threshold = self.filter(abs(noise))
            oc = abs(noise)/threshold.real
            oc.reshape(-1).sort()
            self.factor = oc[-100]
            np.save(path,self.factor)

    def get_threshold(self, data):
        return {'CA':self.get_CA_threshold,
                'SO':self.get_SO_threshold,
                'GO':self.get_GO_threshold
                }[self.CFARType](data)

    def get_CA_threshold(self,data):
        # win = self.winset[::-1].T[::-1].T
        win = self.winset.flip(0,1)
        N,M = win.shape[0]+data.shape[0]-1,win.shape[1]+data.shape[1]-1
        result = torch.fft.ifft2(torch.fft.fft2(data,s=(N,M))*torch.fft.fft2(win,s=(N,M)))
        cout  = torch.fft.ifft2(torch.fft.fft2(torch.ones_like(data),s=(N,M))*torch.fft.fft2(win,s=(N,M)))
        result /= cout
        return result[int(win.shape[0]/2):int(win.shape[0]/2)+data.shape[0],int(win.shape[1]/2):int(win.shape[1]/2)+data.shape[1]]
    
    def get_SO_threshold(self,data):
        threshold = np.zeros_like(data).astype(float)
        W,H = data.shape
        for wi in range(W):
            for hi in range(H):
                wl = 0 if wi-self.ProtectLen-self.ReferenceLen<0 else wi-self.ProtectLen-self.ReferenceLen
                wpl= 0 if wi-self.ProtectLen<0 else wi-self.ProtectLen
                wr = W if wi+self.ProtectLen+self.ReferenceLen+1>W else wi+self.ProtectLen+self.ReferenceLen+1
                wpr = W if wi+self.ProtectLen+1>W else wi+self.ProtectLen+1
                hl = 0 if hi-self.ProtectLen-self.ReferenceLen<0 else hi-self.ProtectLen-self.ReferenceLen
                hpl = 0 if hi-self.ProtectLen<0 else hi-self.ProtectLen
                hr = H if hi+self.ProtectLen+self.ReferenceLen+1>H else hi+self.ProtectLen+self.ReferenceLen+1
                hpr = H if hi+self.ProtectLen+1>H else hi+self.ProtectLen+1
                
                # 左右的最小值
                row_min = data[np.hstack((range(wl,wpl),range(wpr,wr))).astype(int),hl:hr].min() if len(np.hstack((range(wl,wpl),range(wpr,wr)))) else np.inf
                # 上下的最小值
                column_min = data[wl:wr,np.hstack((range(hl,hpl),range(hpr,hr))).astype(int)].min() if len(np.hstack((range(hl,hpl),range(hpr,hr)))) else np.inf

                threshold[wi,hi] = min(row_min,column_min)
        return threshold
        
    def get_GO_threshold(self,data):
        threshold = np.zeros_like(data).astype(float)
        W,H = data.shape
        for wi in range(W):
            for hi in range(H):
                wl = 0 if wi-self.ProtectLen-self.ReferenceLen<0 else wi-self.ProtectLen-self.ReferenceLen
                wpl= 0 if wi-self.ProtectLen<0 else wi-self.ProtectLen
                wr = W if wi+self.ProtectLen+self.ReferenceLen+1>W else wi+self.ProtectLen+self.ReferenceLen+1
                wpr = W if wi+self.ProtectLen+1>W else wi+self.ProtectLen+1
                hl = 0 if hi-self.ProtectLen-self.ReferenceLen<0 else hi-self.ProtectLen-self.ReferenceLen
                hpl = 0 if hi-self.ProtectLen<0 else hi-self.ProtectLen
                hr = H if hi+self.ProtectLen+self.ReferenceLen+1>H else hi+self.ProtectLen+self.ReferenceLen+1
                hpr = H if hi+self.ProtectLen+1>H else hi+self.ProtectLen+1
                
                row_max = data[np.hstack((range(wl,wpl),range(wpr,wr))).astype(int),hl:hr].max() if len(np.hstack((range(wl,wpl),range(wpr,wr)))) else 0
                column_max = data[wl:wr,np.hstack((range(hl,hpl),range(hpr,hr))).astype(int)].max() if len(np.hstack((range(hl,hpl),range(hpr,hr)))) else 0

                threshold[wi,hi] = max(row_max,column_max)
        return threshold


class CFAR2D_Parallel():
    def __init__(self, cfar_type='CA', alpha=1.2, guard_cells=3, ref_cells=2) -> None:
        super(CFAR2D_Parallel, self).__init__()
        self.cfar_type = cfar_type
        self.alpha = alpha
        self.guard_cells = guard_cells
        self.ref_cells = ref_cells
        self.N = 4 * self.ref_cells * (self.ref_cells+2*self.guard_cells+1)
        # padding order: l-r-t-b
        self.padding = (guard_cells+ref_cells,
                        guard_cells+ref_cells,
                        guard_cells+ref_cells,
                        guard_cells+ref_cells)
        self.zero_padding = nn.ZeroPad2d(self.padding)
        # (1, 2N, 1, 1)
        self.prf = self._get_cfar_prf_grid(rb=ref_cells, gb=guard_cells, N=self.N)
        # print('prf: ', self.prf.view(2*self.N))
    
    def filter(self, x):
        b, c, h, w = x.shape
        p_r = self._get_p_r(b, h, w, self.N)
        # (b, c, h, w) -> (b, c, h+2*pad_h, w+2*pad_w)
        x_pad = self.zero_padding(x)
        # print('x_pad: ', x_pad)
        # sample x_r (b, c, h, w, N) using p_r
        x_r = self._sample_x(x_pad, p_r, self.N)
        # print('x_r: ', x_r)
        # (b, c, h, w)
        threshold = self.get_threshold(x_r)
        # print('threshold: ', threshold)
        return (x >= self.alpha * threshold).int()

    def get_scr(self, x, eps=1e-8):
        """Continuous signal-to-clutter ratio x / local_noise_floor, instead of filter()'s
        hard alpha-thresholded binary decision -- the local noise-floor estimate (get_threshold)
        already exists per-pixel inside filter(), this just returns the ratio instead of
        discarding it. Used to contrast-normalize a native-resolution RD map before downsampling,
        so pooling doesn't inflate background bins' peaks as much as raw magnitude does (see
        kurals/dataset_process/cfar_normalize_dataset.py)."""
        b, c, h, w = x.shape
        p_r = self._get_p_r(b, h, w, self.N)
        x_pad = self.zero_padding(x)
        x_r = self._sample_x(x_pad, p_r, self.N)
        threshold = self.get_threshold(x_r)
        return x / (threshold + eps)
    
    # generating peak receptive field grid
    def _get_cfar_prf_grid(self, rb, gb, N):
        # relative potision of receptive field grid
        h_t = -(rb + gb)
        h_d = rb + gb
        w_l = -(rb + gb)
        w_r = rb + gb
        # width and height of receptive field
        w_prf = (rb + gb) * 2 + 1
        h_prf = (rb + gb) * 2 + 1
        
        prf_x_idx, prf_y_idx = torch.meshgrid(
            torch.arange(h_t, h_d + 1),
            torch.arange(w_l, w_r + 1), indexing='ij')

        # grid_size*2, 1 grid_size = (h_d*2+1) * (w_r*2+1)
        # taking positions clockwise

        # 四个方向的点数是一样的
        prf_xt = prf_x_idx[0:rb, 0:(w_prf - rb)]
        prf_xr = prf_x_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        # prf_xr = prf_x_idx[rb:(h_prf-2*rb + 1), (w_prf-rb):w_prf]
        prf_xd = prf_x_idx[(h_prf-rb):h_prf, rb:w_prf]
        prf_xl = prf_x_idx[rb:h_prf, 0:rb]
        # prf_xl = prf_x_idx[rb:(h_prf-2*rb + 1), 0:rb]
        prf_x = torch.cat([torch.flatten(prf_xt),
                           torch.flatten(prf_xr),
                           torch.flatten(prf_xd),
                           torch.flatten(prf_xl)], 0)
        prf_yt = prf_y_idx[0:rb, 0:(w_prf - rb)]
        prf_yr = prf_y_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        # prf_yr = prf_y_idx[rb:(h_prf-2*rb + 1), (w_prf-rb):w_prf]
        prf_yd = prf_y_idx[(h_prf-rb):h_prf, rb:w_prf]
        prf_yl = prf_y_idx[rb:h_prf, 0:rb]
        # prf_yl = prf_y_idx[rb:(h_prf-2*rb + 1), 0:rb]
        prf_y = torch.cat([torch.flatten(prf_yt),
                           torch.flatten(prf_yr),
                           torch.flatten(prf_yd),
                           torch.flatten(prf_yl)], 0)

        prf = torch.cat([prf_x, prf_y], 0)
        prf = prf.view(1, 2*N, 1, 1)
        return prf
    
    # getting p_c (the center point coords) from the padded grid of input x
    def _get_p_c(self, b, h, w, N):
        # generating pc_grid
        # the order of padding is l-r-t-b
        p_c_x, p_c_y = torch.meshgrid(
            torch.arange(self.padding[2], h+self.padding[2], 1),
            torch.arange(self.padding[0], w+self.padding[0], 1), indexing='ij')
        p_c_x = torch.flatten(p_c_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_c_y = torch.flatten(p_c_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_c: (1, 2N, h, w)
        p_c = torch.cat([p_c_x, p_c_y], 1)
        # (B, 2N, h, w)
        p_c = p_c.repeat(b, 1, 1, 1)
        return p_c
    
    # getting p_r positions from each p_c
    def _get_p_r(self, b, h, w, N):
        # (B, 2N, h, w)
        p_c = self._get_p_c(b, h, w, N)
        p_r = p_c + self.prf
        # p_r = p_c.unsqueeze(1)
        # (B, 2N, h, w)
        return p_r
    
    # sampling x using p_r or p_c
    def _sample_x(self, x_pad, p, N):
        # (b, 2N, h, w) -> (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        b, h, w, _ = p.size()
        # x_pad: shape of (b, c, H, W)
        h_pad = x_pad.size(2)
        w_pad = x_pad.size(3)
        c = x_pad.size(1)
        # (b, c, h_pad*w_pad)
        # strech each spatial channel of x_pad as 1-D vector
        x_pad = x_pad.contiguous().view(b, c, -1)
        # (b, h, w, N)
        # for faster inference
        p = p.long()
        # transform spatial coord of p into the 1-D index
        index = p[..., :N]*w_pad + p[..., N:]
        # index_x = torch.clamp(p[..., :N], 0, h_pad-1)
        # index_y = torch.clamp(p[..., N:], 0, w_pad-1)
        # index = index_x * w_pad + index_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # for faster inference
        # index=index.long()
        x_r = x_pad.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_r
    
    def get_threshold(self, data):
        return {'CA':self.get_CA_threshold,
                'SO':self.get_SO_threshold,
                'GO':self.get_GO_threshold,
                'OS':self.get_OS_threshold
                }[self.cfar_type](data)
    
    def get_CA_threshold(self, data):
        return torch.mean(data, dim=-1)
    
    def get_SO_threshold(self, data):
        mean_t = torch.mean(data[..., 0:self.N//4], dim=-1)
        mean_r = torch.mean(data[..., self.N//4:2*self.N//4], dim=-1)
        mean_d = torch.mean(data[..., 2*self.N//4:3*self.N//4], dim=-1)
        mean_l = torch.mean(data[..., 3*self.N//4:self.N], dim=-1)
        threshold, _ = torch.min(torch.stack([mean_t, mean_r, mean_d, mean_l], dim=-1), dim=-1)
        return threshold
    
    def get_GO_threshold(self, data):
        mean_t = torch.mean(data[..., 0:self.N//4], dim=-1)
        mean_r = torch.mean(data[..., self.N//4:2*self.N//4], dim=-1)
        mean_d = torch.mean(data[..., 2*self.N//4:3*self.N//4], dim=-1)
        mean_l = torch.mean(data[..., 3*self.N//4:self.N], dim=-1)
        threshold, _ = torch.max(torch.stack([mean_t, mean_r, mean_d, mean_l], dim=-1), dim=-1)
        return threshold
    
    def get_OS_threshold(self, data):
        threshold, _ = torch.median(data, dim=-1)
        return threshold

def test_1d():
    det = CFAR_1D(CFARType = 'SO', referenceLen=2,protectLen=1)
    data = np.array([1,2,0,1,0,0,3])#输入data需要是abs
    result,threshold = det.filter(data)
    
    plt.figure
    plt.plot(range(7),data)
    plt.plot(range(7),threshold.real)
    plt.legend(['orign','threshold'])
    plt.show()

def test_2d():
    import time
    det = CFAR_2D('SO',referenceLen=1,protectLen=0,factor=4.3783)
    data = np.random.randn(256, 256)
    # data = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]]) #输入data需要是abs
    time_start=time.time()
    for _ in range(10):
        result,threshold = det.filter(data)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    # print(f'data is {data}')
    # print(f'threshold is {threshold.real}')
    # print(f'result is {result}')

def test_2d_parallel():
    import time
    det = CFAR2D_Parallel('OS', ref_cells=2, guard_cells=3, alpha=1.2)
    torch.manual_seed(0)
    data = torch.rand((1, 1, 4, 4))
    print('data: ', data)
    result = det.filter(data)
    print('result: ', result)

    data = torch.rand((1, 1, 256, 256))
    time_start=time.time()
    for _ in range(100):
        result = det.filter(data)
    time_end=time.time()
    print('total time cost',time_end-time_start,'s')

if __name__ == '__main__':
    # test_1d()
    # test_2d()
    test_2d_parallel()
