import numpy as np
import os
import pickle as pickle
import cv2
import random
import time
from data_provider import DataProvider
import multiprocessing
import multiprocessing.dummy
import argparse

LIMIT = 5000000
image_size = 64
DATA_DIR = 'data/'
TARGET_PACKET_NAME = 'resize_image_target.npy'
FAKE_PACKET_NAME = 'resize_image_fake.npy'

parser = argparse.ArgumentParser(description="dump low res fake/target images from dir")
parser.add_argument("image_dir", help="image directory, can search images recursively")
parser.add_argument("data_name", help="data name for specifying directory under ./data/data_name")
parser.add_argument("target_or_fake", help="images are target (0) or fake (1) images", type=int)
parser.add_argument("augmentation_factor", help="augmentation_factor 1~2", nargs='?', const=1, type=int)

image_suffixs = ["png", "jpg", "bmp", "jpeg"]

def is_image(name):
  for suffix in image_suffixs:
    if os.path.splitext(name)[-1][1:] == suffix:
      return True
  return False

def gather_images(from_dir):
  images = []
  for (dirpath, dirnames, filenames) in os.walk(from_dir):
    for filename in filenames:
      if is_image(filename):
        file_path = os.path.join(dirpath, filename)
        images.append(file_path)
  return images

def generate_data_packet(src_dir, data_name, target_or_fake, augmentation_factor=1):
  print("Preprocessing and augmenting the {} dataset...It may take several minutes...".format(data_name))
  time.sleep(1)
  if augmentation_factor>2:
    print("resize only support augmentation counts up to 2, origin/horizontal flip")
    augmentation_factor = 2
  if target_or_fake==0:
    packet_name = TARGET_PACKET_NAME
  else:
    packet_name = FAKE_PACKET_NAME
  dir_path = os.path.join(DATA_DIR, data_name)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  image_pack_path = os.path.join(DATA_DIR, data_name, packet_name)
  files = gather_images(src_dir)
  print("files: {}".format(len(files)))
  files = sorted(files)
  data = {}
  data['filenames'] = [None for _ in range(len(files))]
  images = np.empty(
      (augmentation_factor * len(files), image_size, image_size, 3),
      dtype=np.float32)

  cores = multiprocessing.cpu_count()//2
  print("cores: {}".format(cores))
  p = multiprocessing.dummy.Pool(cores)

  def load(i):
    image_path = files[i]
    data['filenames'][i] = image_path
    # print('%s, %d / %d' % (image_path, i, len(files)))
    image = (cv2.imread(image_path)[:, :, ::-1] /
            255.0).astype(np.float32)
    resized_image = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA)

    images[i * augmentation_factor + 0] = resized_image

    if augmentation_factor==2:
      images[i * augmentation_factor + 1] = resized_image[:, ::-1, :]

  p.map(load, list(range(len(files))))
  print('Data pre-processing finished. Writing....')
  np.save(image_pack_path, images)
  print()


class ResizeDataProvider(DataProvider):

  def __init__(self, data_name, target_or_fake, *args, **kwargs):
    if target_or_fake == 0:
      packet_name = TARGET_PACKET_NAME
    else:
      packet_name = FAKE_PACKET_NAME
    image_pack_path = os.path.join(DATA_DIR, data_name, packet_name)
    data = np.load(image_pack_path)
    print(("#image pack", len(data)))

    super(ResizeDataProvider, self).__init__(data, *args, **kwargs)


def test(data_name, target_or_fake):
  dp = ResizeDataProvider(data_name, target_or_fake)
  d = dp.get_next_batch(10)
  # cv2.imshow('img', d[0][0, :, :, ::-1])
  for i in range(10):
    if target_or_fake==0:
      save_image_name = "target_{}.png".format(i)
    else:
      save_image_name = "fake_{}.png".format(i)
    cv2.imwrite(save_image_name, (d[0][i, :, :, ::-1]*255).astype(np.int32))


if __name__ == '__main__':
  args = parser.parse_args()
  src_dir = args.image_dir
  data_name = args.data_name
  target_or_fake = args.target_or_fake
  augmentation_factor = args.augmentation_factor
  generate_data_packet(src_dir, data_name, target_or_fake, augmentation_factor)
  test(data_name, target_or_fake)
