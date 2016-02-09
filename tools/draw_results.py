#!/usr/bin/python

from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
import numpy as np
import glob
import os
import re
import pandas as pd


# Constants
kScoreTol = 0.5


# Paths
root_dir = '/afs/cs.stanford.edu/u/mhsung/home/data/test_nyud2_custom/'
color_image_dir = root_dir + '/data/Colors/'
depth_image_dir = root_dir + '/data/depth/'
result_dir = root_dir + '/part_results/test/'
output_box_dir = result_dir + '/boxes/'
output_image_dir = result_dir + '/images/'
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


# HSV values in [0..1]
# returns [r, g, b] values from 0 to 255
def hsv_to_rgb(h, s, v):
  h_i = int(h*6)
  f = h*6 - h_i
  p = v * (1 - s)
  q = v * (1 - f*s)
  t = v * (1 - (1 - f) * s)
  if h_i == 0: r, g, b = v, t, p
  if h_i == 1: r, g, b = q, v, p
  if h_i == 2: r, g, b = p, v, t
  if h_i == 3: r, g, b = p, q, v
  if h_i == 4: r, g, b = t, p, v
  if h_i == 5: r, g, b = v, p, q
  return [int(r * 255), int(g * 255), int(b * 255)]


def random_color(index):
    # use golden ratio
    golden_ratio_conjugate = 0.618033988749895
    h = 1
    h += index * golden_ratio_conjugate
    h %= 1
    return hsv_to_rgb(h, 0.95, 0.95)


def draw_rect_image(im, xmin, ymin, xmax, ymax, text, color):
    width, height = im.size
    assert(xmin >= 0)
    assert(ymin >= 0)
    assert(xmax <= width)
    assert(ymax <= height)

    dr = ImageDraw.Draw(im)
    dr.rectangle(((xmin, ymin), (xmax, ymax)),\
            outline=(color[0], color[1], color[2]))
    fo = ImageFont.truetype(font_path, 16)
    dr.text((xmin, ymin), text, (255, 255, 255), font=fo)


""" main """
re_test = re.compile('(\w+)_test_(\w+)')
result_files = glob.glob(result_dir + '/*.txt')

results = dict()
for result_file in result_files:
    result_filename = os.path.splitext(os.path.basename(result_file))[0]
    re_match = re_test.search(result_filename)
    class_name = re_match.group(2)
    results[class_name] = pd.read_csv(result_file, sep=' ', header=None,\
            names=['input_filename', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])



if not os.path.isdir(output_box_dir):
    os.makedirs(output_box_dir)

if not os.path.isdir(output_image_dir):
    os.makedirs(output_image_dir)


image_files = glob.glob(color_image_dir + '/*.png')
for image_file in image_files:
    input_filename = os.path.splitext(os.path.basename(image_file))[0]
    print(input_filename)

    color_im = Image.open(color_image_dir + '/' + input_filename + '.png')
    depth_im = Image.open(depth_image_dir + '/' + input_filename + '.png')

    # Rescale the 16bit depth image and convert the gratscale image to a rgb image
    max_depth = depth_im.getextrema()[1]
    temp_im = Image.new('L', depth_im.size)
    temp_im.putdata(depth_im.getdata(), (255.0 / max_depth))
    depth_im = Image.merge('RGB', [temp_im, temp_im, temp_im])

    out_box_filename = output_box_dir + '/' + input_filename + '.csv'
    box_file = open(out_box_filename, 'w')

    for class_idx, class_name in enumerate(results):
        class_results = results[class_name]
        color = random_color(class_idx)

        image_results = class_results.loc[class_results['input_filename'] == input_filename]
        for box_idx, row in image_results.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            score = row['score']
            box_file.write("%f,%f,%f,%f,%d,%f\n" % \
                    (xmin, ymin, xmax, ymax, class_idx, score));

            if score >= kScoreTol:
                text = class_name + '[' + str(score) + ']'
                draw_rect_image(color_im, xmin, ymin, xmax, ymax, text, color)
                draw_rect_image(depth_im, xmin, ymin, xmax, ymax, text, color)

    box_file.close()

    width, height = color_im.size
    im = Image.new('RGB', (2 * width, height))
    im.paste(color_im, (0, 0))
    im.paste(depth_im, (width, 0))
    out_input_filename = output_image_dir + '/' + input_filename + '.png'
    im.save(out_input_filename)

