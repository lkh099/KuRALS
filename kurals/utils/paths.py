"""Class to get global paths"""
from pathlib import Path
from kurals.utils import KURAD_HOME
from kurals.utils.configurable import Configurable


class Paths(Configurable):

    def __init__(self):
        self.config_path = KURAD_HOME / 'config_files' / 'config.ini'
        super().__init__(self.config_path)
        self.paths = dict()
        self._build()

    def _build(self):
        # warehouse = Path(self.config['data']['warehouse'])
        # self.paths['warehouse'] = warehouse
        self.paths['logs'] = Path(self.config['data']['logs'])
        self.paths['KuRALS_CW'] = Path(self.config['data']['cwr'])
        self.paths['KuRALS_PD'] = Path(self.config['data']['pdr'])

    def get(self):
        return self.paths
