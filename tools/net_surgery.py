#!/usr/bin/python

import os.path as osp
import sys

this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '../../', 'caffe-fast-rcnn', 'python')
sys.path.insert(0, caffe_path)

lib_path = osp.join(this_dir, '../../', 'lib')
sys.path.insert(0, lib_path)

import caffe
import roi_data_layer


to_net = caffe.Net('train.prototxt', 'template.caffemodel', caffe.TRAIN)
from_net = caffe.Net('../VGG16/fast_rcnn/train.prototxt', '../../data/fast_rcnn_models/vgg16_fast_rcnn_iter_40000.caffemodel', caffe.TRAIN)

to_pr = []
from_pr = []
for key, value in to_net.params.iteritems():
    if key.find('cls_score') >= 0 or key.find('bbox_pred') >= 0:
        continue
    print key
    to_pr.append(key)
    from_pr.append(key)

to_params = {pr: (to_net.params[pr][0].data, to_net.params[pr][1].data) \
        for pr in to_pr}
from_params = {pr: (from_net.params[pr][0].data, from_net.params[pr][1].data) \
        for pr in from_pr}

for to_key, from_key in zip(to_pr, from_pr):
    to_params[to_key][0].flat = from_params[from_key][0].flat  # flat unrolls the arrays
    to_params[to_key][1][...] = from_params[from_key][1]

to_net.save('transplanted.caffemodel')

