"""Script to set the path to CARRADA in the config.ini file"""
import argparse
from kurals.utils.configurable import Configurable
from kurals.utils import KURAD_HOME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings paths for training and testing.')
    parser.add_argument('--cwr', default='/datasets_cwr',
                        help='Path to the continuous wave radar dataset.')
    parser.add_argument('--pdr', default='/datasets_pdr',
                        help='Path to the pulse doppler radar dataset.')
    parser.add_argument('--logs', default='/root/workspace/logs',
                        help='Path to the save the logs and models.')
    args = parser.parse_args()
    config_path = KURAD_HOME / 'config_files' / 'config.ini'
    configurable = Configurable(config_path)
    configurable.set('data', 'cwr', args.cwr)
    configurable.set('data', 'pdr', args.pdr)
    configurable.set('data', 'logs', args.logs)
    with open(config_path, 'w') as fp:
        configurable.config.write(fp)
