import sys
import os
from util import load_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from net import GAN

TRAIN = 0
EVAL = 1
RESTORE_TRAIN = 2

def main():
  config_name = sys.argv[1]
  cfg = load_config(config_name)
  cfg.name = sys.argv[1] + '/' + sys.argv[2]
  net = GAN(cfg, mode=RESTORE_TRAIN)
  net.train()


if __name__ == '__main__':
  main()
