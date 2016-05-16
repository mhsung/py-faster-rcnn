#!/usr/bin/python

# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import scipy.sparse
import scipy.io as sio


devkit_path = "/afs/cs.stanford.edu/u/mhsung/home/data/rcnn_results/object_synthesis/"

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def load_custom_annotation(imagename, classidx):
    """
    Load image and bounding boxes info from mat files.
    """
    filename = os.path.join(devkit_path, 'data', 'annotations', imagename + '.mat')
    # print 'Loading: {}'.format(filename)

    assert os.path.exists(filename), \
        'Custom data not found at: {}'.format(filename)
    # input format: [ymin xmin ymax xmax gt_cls (gt_cls_overlap)]
    objs = sio.loadmat(filename)['boxes']
    num_objs = objs.shape[0]
    assert (objs.shape[1] >= 5)

    sub_objs = objs[[x for x in range(num_objs) if objs[x, 4] == classidx], :]
    num_sub_objs = sub_objs.shape[0]

    # boxes: [xmin ymin xmax ymax]
    bbox = (sub_objs[:, (1, 0, 3, 2)] - 1).astype(np.int32)
    #gt_classes = (sub_objs[:, 4]).astype(np.int32)
    det = [False] * num_sub_objs

    return {'bbox': bbox,
            'num_objs': num_sub_objs,
            "det": det}

def voc_eval(detpath,
             imagesetfile,
             classidx,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        #if imagename != "test_scene_2_3_16": continue
        class_recs[imagename] = load_custom_annotation(imagename, classidx)
        npos = npos + class_recs[imagename]['num_objs']

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]

    #
    splitlines = [x for x in splitlines if x[0] in imagenames]
    #

    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        #if image_ids[d] != "test_scene_2_3_16": continue
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            #if not R['difficult'][jmax]:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    print("# GTs = " + str(npos))
    print("# Predictions = " + str(nd));
    print("False Positive = " + str(fp[-1]))
    print("True Positive = " + str(tp[-1]))
    print("Recall = " + str(rec[-1]))
    print("Precision = " + str(prec[-1]))
    print("AP = " + str(ap))

    return rec, prec, ap

if __name__ == "__main__":
    detpath_root = "/afs/cs.stanford.edu/u/mhsung/home/app/scene-completion/report/"

    print(os.path.join(devkit_path, 'labels.txt'))
    with open(os.path.join(devkit_path, 'labels.txt'), 'r') as f:
        labels = f.read().splitlines()
        labels.insert(0, '__background__')
        classes = tuple(labels)
        print(classes)

    imagesetfile = os.path.join(detpath_root, '../python', 'test_scene_list.txt')
    print(imagesetfile)

    for cid in range(1, len(classes)):
        classname = classes[cid]
        print("[" + str(cid) + "]: " + classname)
        detpath = os.path.join(devkit_path, 'results', 'det_test_{0}.txt')
        #detpath = os.path.join(detpath_root, 'prediction_{0}.txt')
        #detpath = os.path.join(detpath_root, 'retrieval_{0}.txt')
        voc_eval(detpath, imagesetfile, cid, classname, 0.3)

