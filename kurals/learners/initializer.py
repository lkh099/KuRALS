"""Initializer class to prepare training"""
import json
from torch.utils.data import DataLoader

from kurals.utils.paths import Paths
from kurals.loaders.dataset import KuRALS_CW
from kurals.loaders.dataset import KuRALS_PD
from kurals.loaders.dataloaders import SequenceDataset
from kurals.utils.distributed_utils import get_rank
import time


class Initializer:
    """Class to prepare training model

    PARAMETERS
    ----------
    cfg: dict
        Configuration file used for train/test
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = Paths().get()

    def _get_data(self):
        if self.cfg['dataset'] == 'KuRALS_CW':
            data = KuRALS_CW()
        elif self.cfg['dataset'] == 'KuRALS_PD':
            data = KuRALS_PD()
        else:
            raise KeyError('Dataset {} has not been supported yet.'.format(self.cfg['dataset']))
        
        train = data.get('Train')
        val = data.get('Validation')
        test = data.get('Test')
        return [train, val, test]

    def _get_datasets(self):
        data = self._get_data()
        trainset = SequenceDataset(data[0])
        valset = SequenceDataset(data[1])
        testset = SequenceDataset(data[2])
        return [trainset, valset, testset]

    def _get_dataloaders(self):
        trainset, valset, testset = self._get_datasets()
        trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        return [trainloader, valloader, testloader]

    def _structure_data(self):
        data = dict()
        dataloaders = self._get_dataloaders()
        name_exp = (self.cfg['model'] + '_' +
                    'e' + str(self.cfg['nb_epochs']) + '_' +
                    'lr' + str(self.cfg['lr']) + '_' +
                    's' + str(self.cfg['torch_seed']))
        self.cfg['name_exp'] = name_exp
        folder_path = self.paths['logs'] / self.cfg['dataset'] / self.cfg['model'] / name_exp

        temp_folder_name = folder_path.name + '_' + str(self.cfg['version']) + '_' + time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        temp_folder_path = folder_path.parent / temp_folder_name
        while temp_folder_path.exists():
            self.cfg['version'] += 1
            temp_folder_name = folder_path.name + '_' + str(self.cfg['version']) + '_' + time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
            temp_folder_path = folder_path.parent / temp_folder_name
        folder_path = temp_folder_path

        self.paths['results'] = folder_path / 'results'
        self.paths['writer'] = folder_path / 'boards'
        if get_rank() == 0:
            self.paths['results'].mkdir(parents=True, exist_ok=True)
            self.paths['writer'].mkdir(parents=True, exist_ok=True)

            config_path = folder_path / 'config.json'
            with open(config_path, 'w') as fp:
                json.dump(self.cfg, fp)

        data['cfg'] = self.cfg
        data['paths'] = self.paths
        data['dataloaders'] = dataloaders
        return data

    def get_data(self):
        """Return parameters of the training"""
        return self._structure_data()
