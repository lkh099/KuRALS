"""Class to get global paths"""
import os
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
        # Per-process overrides so concurrent runs can point at different dataset
        # resolutions without racing on the single shared config.ini.
        if os.environ.get('KURALS_CWR'):
            self.paths['KuRALS_CW'] = Path(os.environ['KURALS_CWR'])
        if os.environ.get('KURALS_LOGS'):
            self.paths['logs'] = Path(os.environ['KURALS_LOGS'])

    def get(self):
        return self.paths
