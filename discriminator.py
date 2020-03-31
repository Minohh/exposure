import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from util import load_config, Tee
import shutil
import numpy as np
import tensorflow as tf
from net import GAN
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

device = '/gpu:0'

TRAIN = 0
EVAL = 1
RESTORE_TRAIN = 2

# A small part of this script is based on https://github.com/Zardinality/WGAN-tensorflow

def sigmoid(x):
  return 1/(1 + np.exp(-x)) 

def rgb2lum(image):
  image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :, 1] + 0.06 * image[:, :, :, 2]
  return image[:, :, :, None]

def lerp(a, b, l):
  return (1 - l) * a + l * b

def tanh01(x):
  return np.tanh(x) * 0.5 + 0.5

# def tanh_range(l, r, initial=None):

#   def get_activation(left, right, initial):

#     def activation(x):
#       if initial is not None:
#         bias = math.atanh(2 * (initial - left) / (right - left) - 1)
#       else:
#         bias = 0
#       return tanh01(x + bias) * (right - left) + left
#     return activation

#   return get_activation(l, r, initial)

def tanh_range(l, r, initial=None):

  def get_activation(left, right, initial):

    def activation(x):
      return x
    return activation

  return get_activation(l, r, initial)

def rgb2hsv(rgb):
  return mpl.colors.rgb_to_hsv(rgb)

def hsv2rgb(hsv):
  return mpl.colors.hsv_to_rgb(hsv)

class Filter:

  def __init__(self, net, cfg):
    self.cfg = cfg

    # Specified in child classes
    self.num_filter_parameters = None
    self.short_name = None
    self.filter_parameters = None

  # Should be implemented in child classes
  def filter_param_regressor(self, features):
    assert False

  # Process the whole image, without masking
  # Should be implemented in child classes
  def process(self, img, param):
    assert False

  # Apply the whole filter with masking
  def apply(self,
            img,
            specified_parameter=None):
    assert (specified_parameter is not None)
    filter_parameters = self.filter_param_regressor(specified_parameter)
    low_res_output = self.process(img, filter_parameters)
    return low_res_output

class ExposureFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tanh_range(
        -self.cfg.exposure_range, self.cfg.exposure_range, initial=0)(features)

  def process(self, img, param):
    return img * np.exp(param[:, None, None, :] * np.log(2))

class GammaFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    log_gamma_range = np.log(self.cfg.gamma_range)
    return np.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

  def process(self, img, param):
    return np.maximum(img, 0.001)**param[:, None, None, :]

class ImprovedWhiteBalanceFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.channels = 3
    self.num_filter_parameters = self.channels

  def filter_param_regressor(self, features):
    log_wb_range = 0.5
    mask = np.array(((0, 1, 1)), dtype=np.float32).reshape(1, 3)
    print(mask.shape)
    assert mask.shape == (1, 3)
    features = features * mask
    color_scaling = np.exp(tanh_range(-log_wb_range, log_wb_range)(features))
    # There will be no division by zero here unless the WB range lower bound is 0
    # normalize by luminance
    color_scaling *= 1.0 / (
        1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
        0.06 * color_scaling[:, 2])[:, None]
    return color_scaling

  def process(self, img, param):
    return img * param[:, None, None, :]

class ColorFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.curve_steps = cfg.curve_steps
    self.channels = int(net.shape[3])
    self.num_filter_parameters = self.channels * cfg.curve_steps

  def filter_param_regressor(self, features):
    color_curve = np.reshape(
        features, (-1, self.channels,
                         self.cfg.curve_steps))[:, None, None, :]
    color_curve = tanh_range(
        *self.cfg.color_curve_range, initial=1)(color_curve)
    return color_curve

  def process(self, img, param):
    color_curve = param
    # There will be no division by zero here unless the color filter range lower bound is 0
    color_curve_sum = np.sum(param, axis=4) + 1e-30
    total_image = img * 0
    for i in range(self.cfg.curve_steps):
      total_image += np.clip(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) * \
                     color_curve[:, :, :, :, i]
    total_image *= self.cfg.curve_steps / color_curve_sum
    return total_image

class ToneFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.curve_steps = cfg.curve_steps
    self.num_filter_parameters = cfg.curve_steps

  def filter_param_regressor(self, features):
    tone_curve = np.reshape(
        features, (-1, 1, self.cfg.curve_steps))[:, None, None, :]
    tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
    return tone_curve

  def process(self, img, param):
    # img = tf.minimum(img, 1.0)
    tone_curve = param
    tone_curve_sum = np.sum(tone_curve, axis=4) + 1e-30
    total_image = img * 0
    for i in range(self.cfg.curve_steps):
      total_image += np.clip(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
                     * param[:, :, :, :, i]
    total_image *= self.cfg.curve_steps / tone_curve_sum
    img = total_image
    return img

class ContrastFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    # return tf.sigmoid(features)
    return np.tanh(features)

  def process(self, img, param):
    luminance = np.minimum(np.maximum(rgb2lum(img), 0.0), 1.0)
    contrast_lum = -np.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    return lerp(img, contrast_image, param[:, :, None, None])

class WNBFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return sigmoid(features)

  def process(self, img, param):
    luminance = rgb2lum(img)
    return lerp(img, luminance, param[:, :, None, None])

class SaturationPlusFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return sigmoid(features)

  def process(self, img, param):
    img = np.minimum(img, 1.0)
    hsv = rgb2hsv(img)
    s = hsv[:, :, :, 1:2]
    v = hsv[:, :, :, 2:3]
    # enhanced_s = s + (1 - s) * 0.7 * (0.5 - tf.abs(0.5 - v)) ** 2
    enhanced_s = s + (1 - s) * (0.5 - np.abs(0.5 - v)) * 0.8
    hsv1 = np.concatenate([hsv[:, :, :, 0:1], enhanced_s, hsv[:, :, :, 2:]], axis=3)
    full_color = hsv2rgb(hsv1)

    param = param[:, :, None, None]
    color_param = param
    img_param = 1.0 - param

    return img * img_param + full_color * color_param

class Discriminator_eval(GAN):
  def __init__(self, cfg):
    super().__init__(cfg, mode=EVAL)

    N = self.cfg.batch_size
    exposure_parameter = np.repeat(np.array([[-1]]), N, axis=0)
    gamma_parameter = np.repeat(np.array([[1.1]]), N, axis=0)
    wb_parameter = np.repeat(np.array([[1.1, 1.1, 1.1]]), N, axis=0)
    saturation_parameter = np.repeat(np.array([[10]]), N, axis=0)
    tone_parameter = np.repeat(np.array([[10, 1, 1, 1, 1, 10, 10, 10]]), N, axis=0)
    contrast_parameter = np.repeat(np.array([[8]]), N, axis=0)
    wnb_parameter = np.repeat(np.array([[.5]]), N, axis=0)
    color_parameter = np.repeat(np.repeat(np.array([[1, 1, 1, 10, 1, 1, 10, 10]]), 3, axis=0), N, axis=0)
    
    self.specified_parameters = [exposure_parameter,
                                 gamma_parameter,
                                 wb_parameter,
                                 saturation_parameter,
                                 tone_parameter,
                                 contrast_parameter,
                                 wnb_parameter,
                                 color_parameter]
    self.filters = [
        ExposureFilter, GammaFilter, ImprovedWhiteBalanceFilter,
        SaturationPlusFilter, ToneFilter, ContrastFilter, WNBFilter, ColorFilter
    ]

    self.tee = Tee(os.path.join(self.dir, 'discriminator.txt'))

  def eval(self):
    self.real_dataset = self.cfg.real_data_provider()
    indices = np.arange(self.cfg.batch_size)*500
    real_data_batch = self.real_dataset.data[indices.tolist()]

    filters = [x(real_data_batch, self.cfg) for x in self.filters]
    real_logits = []
    fake_logits = []
    emds = []
    imgs = []
    for j, filter in enumerate(filters):
      filtered_image_batch = filter.apply(
          real_data_batch, specified_parameter=self.specified_parameters[j])
      feed_dict = {
          self.real_data: real_data_batch,
          self.fake_output: filtered_image_batch,
          self.is_train: 0
      }

      real_logit, fake_logit, emd = self.sess.run(
          [
              self.real_logit, self.fake_logit, self.emd
          ],
          feed_dict=feed_dict)
      real_logits.append(real_logit.tolist())
      fake_logits.append(fake_logit.tolist())
      emds.append(emd.tolist())
      imgs.append((filtered_image_batch*255).astype(np.uint8))

    print("EMD = ", emds)
    print(real_logits)
    print(fake_logits)
    org_imgs = (real_data_batch*255).astype(np.uint8)
    self.save_and_print(org_imgs, imgs, emds)

    return

  def save_and_print(self, org_imgs ,imgs, emds):
    fig = plt.figure(figsize=(9, 9), dpi=80)
    ax = fig.subplots(9, 9)
    
    ax[0][0].axis('off')
    ax[0][0].text(.3,.3, 'EMD')

    for i in range(len(self.filters)):
      ax[i+1][0].axis('off')
      ax[i+1][0].text(0, .3, "{:.4f}".format(emds[i]))
    for j in range(self.cfg.batch_size):
      if j % 8 == 0:
        ax[0][j//8+1].axis('off')
        ax[0][j//8+1].imshow(org_imgs[j])

    for i in range(len(self.filters)):
      for j in range(self.cfg.batch_size):
        if j % 8 == 0:
          ax[i+1][j//8+1].axis('off')
          ax[i+1][j//8+1].imshow(imgs[i][j])
    fig.savefig(os.path.join(self.dir, 'filters.png'))

def evaluate():
  if len(sys.argv) < 3:
    print(
        "Usage: python3 evaluate.py [config suffix] [model name]"
    )
    exit(-1)
  if len(sys.argv) == 3:
    print(
        " Note: Process a single image at a time may be inefficient - try multiple inputs)"
    )
  print("(TODO: batch processing when images have the same resolution)")
  print()
  print("Initializing...")
  config_name = sys.argv[1]
  import shutil
  shutil.copy('models/%s/%s/scripts/config_%s.py' %
              (config_name, sys.argv[2], config_name), 'config_tmp.py')
  cfg = load_config('tmp')
  cfg.name = sys.argv[1] + '/' + sys.argv[2]
  net = Discriminator_eval(cfg)
  net.eval()

if __name__ == '__main__':
  evaluate()
