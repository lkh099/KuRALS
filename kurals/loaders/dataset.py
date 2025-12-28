"""Class to load the CWR dataset"""
import json
from kurals.utils.paths import Paths
import numpy as np


class KuRALS_CW:
    """Class to load KuRALS_CW dataset"""

    def __init__(self):
        self.paths = Paths().get()
        # self.warehouse = self.paths['warehouse']
        self.kurals_cw = self.paths['KuRALS_CW']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self.ratio_spilt = [0.8, 0.1, 0.1]
        self._split()

    def _load_data_seq_ref(self):
        path = self.kurals_cw / 'data_seq_ref.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.kurals_cw / 'light_dataset_frame_oriented.json'
        with open(path, 'r', encoding='GBK') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        cum_ratio = np.cumsum(self.ratio_spilt)
        for sequence in self.annotations.keys():
            if '机场人车' not in sequence:
                num_frames = len(self.annotations[sequence])
                # self.annotations[sequence] = np.random.permutation(num_frames)
                self.train[sequence] = self.annotations[sequence][0:int(num_frames*cum_ratio[0])]
                self.validation[sequence] = self.annotations[sequence]\
                        [int(num_frames*cum_ratio[0]):int(num_frames*cum_ratio[1])]
                self.test[sequence] = self.annotations[sequence][int(num_frames*cum_ratio[1]):]
            else:
                num_frames = len(self.annotations[sequence]) - 200
                self.train[sequence] = self.annotations[sequence][0:int(200*cum_ratio[0])] + \
                    self.annotations[sequence][200:200+int(num_frames*cum_ratio[0])]

                self.validation[sequence] = self.annotations[sequence][int(200*cum_ratio[0]):int(200*cum_ratio[1])] +\
                    self.annotations[sequence][200+int(num_frames*cum_ratio[0]):200+int(num_frames*cum_ratio[1])]

                self.test[sequence] = self.annotations[sequence][int(200*cum_ratio[1]):200]+\
                    self.annotations[sequence][200+int(num_frames*cum_ratio[1]):]

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))

class KuRALS_PD:
    """Class to load KuRALS_PD dataset"""

    def __init__(self):
        self.paths = Paths().get()
        # self.warehouse = self.paths['warehouse']
        self.kurals_pd = self.paths['KuRALS_PD']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self.ratio_spilt = [0.8, 0.1, 0.1]
        self._split()

    def _load_data_seq_ref(self):
        path = self.kurals_pd / 'data_seq_ref.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.kurals_pd / 'light_dataset_frame_oriented.json'
        with open(path, 'r', encoding='GBK') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        cum_ratio = np.cumsum(self.ratio_spilt)
        for sequence in self.annotations.keys():
            num_frames = len(self.annotations[sequence])
            # self.annotations[sequence] = np.random.permutation(num_frames)
            self.train[sequence] = self.annotations[sequence][0:int(num_frames*cum_ratio[0])]
            self.validation[sequence] = self.annotations[sequence]\
                    [int(num_frames*cum_ratio[0]):int(num_frames*cum_ratio[1])]
            self.test[sequence] = self.annotations[sequence][int(num_frames*cum_ratio[1]):]

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))

def test_cwr():
    """Method to test the cwrdata dataset"""
    dataset = KuRALS_CW().get('Train')
    print(dataset['2020年11月11日15时50分44秒_mtd_mid'])
    
def test_pdr():
    """Method to test the pdrdata dataset"""
    dataset = KuRALS_PD().get('Train')
    print(dataset['20191210_145742_Ku5km_SeaBoat'])

if __name__ == '__main__':
    test_pdr()
