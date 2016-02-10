#!/usr/bin/python

from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
import numpy as np

# NOTE:
# Set a specific font path on your machine
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

