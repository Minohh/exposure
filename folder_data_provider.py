import numpy as np
import os
import cv2
import random
from util import get_image_center
from data_provider import DataProvider
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

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

def preprocess_images(image_paths):
  data = []
  for image_path in image_paths:
    image = (cv2.imread(image_path)[:, :, ::-1] /
            255.0).astype(np.float32)
    # change to resize the whole image
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    data.append(image)
    # data.append(image[:, ::-1, :])
  return data

class FolderDataProvider(DataProvider):

  def __init__(self,
               folder,
               read_limit=-1,
               main_size=80,
               crop_size=64,
               augmentation_factor=4,
               *args,
               **kwargs):
    # files = os.listdir(folder)
    files = gather_images(folder)
    print(folder)
    print(files[0])
    files = sorted(files)

    if read_limit != -1:
      files = files[:read_limit]
    data = []
    files.sort()

    cores = multiprocessing.cpu_count()
    print("cores: {}".format(cores))
    file_count = len(files)
    work_file_count = int(file_count/cores)
    with ThreadPoolExecutor(max_workers=cores) as executor:
      start = 0
      end = work_file_count
      futures = []
      for i in range(cores):
        if i==(cores-1):
          end = file_count
        future = executor.submit(preprocess_images, files[start:end])
        futures.append(future)
        start += work_file_count
        end += work_file_count
      for i in range(cores):
        data.extend(futures[i].result())
    # for f in files:
      # image = (cv2.imread(image_paths)[:, :, ::-1] /
      #       255.0).astype(np.float32)
      # change to resize the whole image
      # image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
      # data.append(image)
      # data.append(image[:, ::-1, :])
      # center image
      # image = get_image_center(image)
      # image = cv2.resize(
      #     image, (main_size, main_size), interpolation=cv2.INTER_AREA)
      # for i in range(augmentation_factor):
      #   new_image = image
      #   if random.random() < 0.5:
      #     new_image = new_image[:, ::-1, :]
      #   sx, sy = random.randrange(main_size - crop_size + 1), random.randrange(
      #       main_size - crop_size + 1)
      #   data.append(new_image[sx:sx + crop_size, sy:sy + crop_size])
    data = np.stack(data, axis=0)
    print("# image after augmentation =", len(data))
    super(FolderDataProvider, self).__init__(data, *args, bnw=False,
                                             augmentation=1.0,
                                             output_size=crop_size,
                                             **kwargs)

def test():
  dp = FolderDataProvider('data/sintel/outputs')
  while True:
    d = dp.get_next_batch(64)
    cv2.imshow('img', d[0][0, :, :, ::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
  test()
  # preprocess()
