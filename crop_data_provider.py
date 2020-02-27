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
TARGET_PACKET_NAME = 'crop_image_target.npy'
FAKE_PACKET_NAME = 'crop_image_fake.npy'


parser = argparse.ArgumentParser(description="dump low res fake/target images from dir")
parser.add_argument("image_dir", help="image directory, can search images recursively")
parser.add_argument("data_name", help="data name for specifying directory under ./data/data_name")
parser.add_argument("target_or_fake", help="images are target (0) or fake (1) images")

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
  if target_or_fake==0:
    packet_name = TARGET_PACKET_NAME
  else:
    packet_name = FAKE_PACKET_NAME
  dir_path = os.path.join(DATA_DIR, data_name)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  image_pack_path = os.path.join(DATA_DIR, data_name, packet_name)
  files = gather_images(src_dir)
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

    h = image.shape[0]
    w = image.shape[1]
    shorter_edge = min(w, h)

    # the first image should always be hand-selected
    # top square for vertical screen
    # median square for horizontal screen
    if w > h:
      sx = (w - h)//2
      sy = 0
    else:
      sx = 0
      sy = 0
    new_image = image[sy:sy+shorter_edge, sx:sx+shorter_edge]
    images[i * augmentation_factor + 0] = cv2.resize(
            new_image,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA)

    # Crop some patches so that non-square images are better covered
    for j in range(1, augmentation_factor):
      sx = random.randrange(0, w - shorter_edge + 1)
      sy = random.randrange(0, h - shorter_edge + 1)
      new_image = image[sy:sy+shorter_edge, sx:sx+shorter_edge]
      images[i * augmentation_factor + j] = cv2.resize(
          new_image,
          dsize=(image_size, image_size),
          interpolation=cv2.INTER_AREA)

  p.map(load, list(range(len(files))))
  print('Data pre-processing finished. Writing....')
  np.save(image_pack_path, images)
  print()


class CropDataProvider(DataProvider):

  def __init__(self, data_name, target_or_fake, *args, **kwargs):
    if target_or_fake == 0:
      packet_name = TARGET_PACKET_NAME
    else:
      packet_name = FAKE_PACKET_NAME
    image_pack_path = os.path.join(DATA_DIR, data_name, packet_name)
    data = np.load(image_pack_path)
    print(("#image pack", len(data)))

    super(CropDataProvider, self).__init__(data, *args, **kwargs)


def test(data_name, target_or_fake):
  dp = CropDataProvider(data_name, target_or_fake)
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
  target_or_fake = int(args.target_or_fake)
  print("target_or_fake ", target_or_fake, type(target_or_fake))
  generate_data_packet(src_dir, data_name, target_or_fake)
  test(data_name, target_or_fake)
