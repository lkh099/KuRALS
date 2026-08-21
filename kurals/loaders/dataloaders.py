"""Classes to load Carrada dataset"""
import os
import re
import numpy as np
from skimage import transform
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from kurals.loaders.dataset import KuRALS_CW
from kurals.utils.paths import Paths

_SCAN_NUM_RE = re.compile(r'MTD_Scan_(\d+)_')


def _scan_num(frame_name):
    """Extract the integer scan number from a 'MTD_Scan_<A>_<B>' frame name,
    or None if it doesn't match (treated as non-contiguous by the caller)."""
    m = _SCAN_NUM_RE.match(frame_name)
    return int(m.group(1)) if m else None


class SequenceDataset(Dataset):
    """DataLoader class for dataset sequences"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.seq_names = list(self.dataset.keys())

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        return seq_name, self.dataset[seq_name]


class KuRALSDataset(Dataset):
    """DataLoader class for Ku Radar sequences
    Load frames, only for semantic segmentation
    Specific to load several frames at the same time (sub sequences)
    Aggregated Tensor Input + Multiple Output

    PARAMETERS
    ----------
    dataset: SequenceCarradaDataset object
    annotation_type: str
        Supported annotations are 'sparse', 'dense'
    path_to_frames: str
        Path to the frames of a given sequence (folder of the sequence)
    process_signal: boolean
        Load signal w/ or w/o processing (power, log transform)
    n_frame: int
        Number of frames used for each sample
    transformations: list of functions
        Preprocessing or data augmentation functions
        Default: None
    add_temp: boolean
        Formating the input tensors as sequences
        Default: False
    """

    # A window of n_frames consecutive *list positions* is not guaranteed to be
    # n_frames consecutive *scan* numbers -- light_dataset_frame_oriented.json's
    # per-sequence frame lists are not always scan-sorted/contiguous (confirmed
    # by manually inspecting box annotations at several large-gap points, e.g.
    # the airport sequence's train-split concatenation seam pairs scan 100
    # (class 2, near range-doppler bin [41,264]) directly with scan 139 (class
    # 3, near bin [6,391]) -- two unrelated targets, not consecutive frames).
    # MAX_SCAN_GAP=6 comes from the corpus-wide histogram of consecutive-pair
    # scan-number deltas across all 9 sequences: gaps of 0/1/2-6 account for
    # ~2563 of ~2577 pairs, then a hard cliff straight to 10/14/20/22/27/42/82
    # (and a handful of large negative deltas at a few sequences' own list
    # starts). Only matters once n_frames > 1 (single-frame windows have no
    # adjacency to violate).
    MAX_SCAN_GAP = 6

    def __init__(self, dataset, annotation_type, path_to_frames, process_signal,
                 n_frames, transformations=None, add_temp=False, flip_expand=False):
        self.dataset = dataset
        self.annotation_type = annotation_type
        self.path_to_frames = Path(path_to_frames)
        self.process_signal = process_signal
        self.n_frames = n_frames
        self.transformations = transformations
        self.add_temp = add_temp
        self.flip_expand = flip_expand
        self.path_to_annots = self.path_to_frames / 'annotations' / self.annotation_type

        if self.n_frames > 1:
            base_indices = [
                start for start in range(len(self.dataset) - self.n_frames + 1)
                if all(
                    _scan_num(self.dataset[start + k][0]) is not None
                    and _scan_num(self.dataset[start + k + 1][0]) is not None
                    and abs(_scan_num(self.dataset[start + k + 1][0])
                            - _scan_num(self.dataset[start + k][0])) <= self.MAX_SCAN_GAP
                    for k in range(self.n_frames - 1)
                )
            ]
        else:
            base_indices = list(range(len(self.dataset)))

        if self.flip_expand:
            # Deterministic full coverage of all 4 flip variants (identity/hflip/vflip/both)
            # per sample per epoch, instead of each __getitem__ call drawing one variant at
            # random -- exhaustive rather than stochastic use of the same hflip/vflip
            # transforms, so every epoch trains on genuinely 4x as many distinct views.
            variants = [(False, False), (True, False), (False, True), (True, True)]
            self.valid_indices = [i for i in base_indices for _ in variants]
            self.flip_variants = [v for _ in base_indices for v in variants]
        else:
            self.valid_indices = base_indices
            self.flip_variants = None

    def transform(self, frame, is_vflip=False, is_hflip=False):
        """
        Method to apply preprocessing / data augmentation functions

        PARAMETERS
        ----------
        frame: dict
            Contains the matrices and the masks on which we want to apply the transformations
        is_vfilp: boolean
            If you want to apply a vertical flip
            Default: False
        is_hfilp: boolean
            If you want to apply a horizontal flip
            Default: False

        RETURNS
        -------
        frame: dict
        """
        if self.transformations is not None:
            for function in self.transformations:
                if isinstance(function, VFlip):
                    if is_vflip:
                        frame = function(frame)
                    else:
                        continue
                if isinstance(function, HFlip):
                    if is_hflip:
                        frame = function(frame)
                    else:
                        continue
                if not isinstance(function, VFlip) and not isinstance(function, HFlip):
                    frame = function(frame)
        return frame

    def __len__(self):
        """Number of scan-contiguous windows per sequence (see valid_indices)"""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if self.flip_expand:
            is_hflip, is_vflip = self.flip_variants[idx]
        idx = self.valid_indices[idx]
        init_frame_name = self.dataset[idx+self.n_frames-1][0]
        frame_names = [self.dataset[f_id][0] for f_id in range(idx, idx+self.n_frames)]
        rd_matrices = list()
        rd_mask = np.load(os.path.join(self.path_to_annots, init_frame_name,
                                       'range_doppler.npy'))
        for frame_name in frame_names:
            rd_matrix = np.load(os.path.join(self.path_to_frames,
                                             'range_doppler_numpy',
                                             frame_name + '.npy'))

            rd_matrices.append(rd_matrix)

        # Apply the same transfo to all representations
        if not self.flip_expand:
            is_vflip = np.random.uniform(0, 1) > 0.5
            is_hflip = np.random.uniform(0, 1) > 0.5

        rd_matrix = np.dstack(rd_matrices)
        rd_matrix = np.rollaxis(rd_matrix, axis=-1)
        rd_frame = {'matrix': rd_matrix, 'mask': rd_mask}
        rd_frame = self.transform(rd_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'],
                                                    axis=self.add_temp)

        frame = {'rd_matrix': rd_frame['matrix'], 'rd_mask': rd_frame['mask']}

        return frame


class Rescale:
    """Rescale the image in a sample to a given size.

    PARAMETERS
    ----------
    output_size: tuple or int
        Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, frame):
        matrix, rd_mask, ra_mask = frame['matrix'], frame['rd_mask'], frame['ra_mask']
        h, w = matrix.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # transform.resize induce a smoothing effect on the values
        # transform only the input data
        matrix = transform.resize(matrix, (matrix.shape[0], new_h, new_w))
        return {'matrix': matrix, 'rd_mask': rd_mask, 'ra_mask': ra_mask}


class Flip:
    """
    Randomly flip the matrix with a proba p
    """

    def __init__(self, proba):
        assert proba <= 1.
        self.proba = proba

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        h_flip_proba = np.random.uniform(0, 1)
        if h_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        v_flip_proba = np.random.uniform(0, 1)
        if v_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        return {'matrix': matrix, 'mask': mask}


class HFlip:
    """
    Randomly horizontal flip the matrix with a proba p
    """

    def __init__(self):
        pass

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        matrix = np.flip(matrix, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
        return {'matrix': matrix, 'mask': mask}


class VFlip:
    """
    Randomly vertical flip the matrix with a proba p
    """

    def __init__(self):
        pass

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        matrix = np.flip(matrix, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()
        return {'matrix': matrix, 'mask': mask}


class GainJitter:
    """Multiply the RD matrix by a random per-sample gain factor, log-uniform in
    [gain_min, gain_max]. Simulates natural SNR/gain variation and discourages the
    model from memorizing exact absolute magnitude values -- mask is untouched, this
    isn't a geometric transform. Called every training step (like Rescale), not
    conditionally like Flip/HFlip/VFlip, so each sample gets a fresh random draw.
    """

    def __init__(self, gain_min=0.7, gain_max=1.4):
        self.gain_min = gain_min
        self.gain_max = gain_max

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        gain = np.exp(np.random.uniform(np.log(self.gain_min), np.log(self.gain_max)))
        return {'matrix': matrix * gain, 'mask': mask}


class NoiseJitter:
    """Multiply the RD matrix by per-pixel noise ~ N(1, std) -- simulates sensor
    noise, discourages memorizing exact per-pixel values. Mask is untouched."""

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        noise = np.random.normal(1.0, self.std, size=matrix.shape)
        return {'matrix': matrix * noise, 'mask': mask}


def test_sequence():
    dataset = KuRALS_CW().get('Train')
    dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                            shuffle=False, num_workers=0)
    for i, data in enumerate(dataloader):
        seq_name, seq = data
        if i == 0:
            seq = [subseq[0] for subseq in seq]
            assert seq_name[0] == '2020年11月11日15时50分44秒_mtd_mid'
            assert 'MTD_Scan_1_27' in seq
        else:
            break


def test_CWRdataset():
    paths = Paths().get()
    n_frames = 3
    dataset = KuRALS_CW().get('Train')
    seq_dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = paths['KuRALS_CW'] / seq_name[0]
        frame_dataloader = DataLoader(KuRALSDataset(seq,
                                                     'dense',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            assert list(frame['rd_matrix'].shape[2:]) == [124, 2048]
            assert frame['rd_matrix'].shape[1] == n_frames
            assert list(frame['rd_mask'].shape[2:]) == [124, 2048]
        break


def test_subflip():
    paths = Paths().get()
    n_frames = 3
    dataset = KuRALS_CW().get('Train')
    seq_dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = paths['KuRALS_CW'] / seq_name[0]
        frame_dataloader = DataLoader(KuRALSDataset(seq,
                                                     'dense',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            rd_matrix = frame['rd_matrix'][0].cpu().detach().numpy()
            rd_mask = frame['rd_mask'][0].cpu().detach().numpy()
            rd_frame_test = {'matrix': rd_matrix,
                             'mask': rd_mask}
            rd_frame_vflip = VFlip()(rd_frame_test)
            rd_matrix_vflip = rd_frame_vflip['matrix']
            rd_frame_hflip = HFlip()(rd_frame_test)
            rd_matrix_hflip = rd_frame_hflip['matrix']
            assert rd_matrix[0][0][0] == rd_matrix_vflip[0][0][-1]
            assert rd_matrix[0][0][-1] == rd_matrix_vflip[0][0][0]
            assert rd_matrix[0][0][0] == rd_matrix_hflip[0][-1][0]
            assert rd_matrix[0][-1][0] == rd_matrix_hflip[0][0][0]
        break

    
if __name__ == '__main__':
    #test_sequence()
    # test_CWRdataset()
    test_subflip()
