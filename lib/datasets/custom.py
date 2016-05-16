# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import custom_utils
import datasets
import datasets.custom
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
from PIL import Image
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess


class custom(datasets.imdb):
    def __init__(self, image_set, devkit_path):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        #self._classes = ('__background__',  # always index 0
        #                 'person')
        # Load label file
        with open(devkit_path + 'labels.txt', 'r') as f:
            labels = f.read().splitlines()
            labels.insert(0, '__background__')
            self._classes = tuple(labels)
            print(self._classes)

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg', '.png']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # Specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000}

        assert os.path.exists(self._devkit_path), \
            'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'images',
                                      index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /image_sets/val.txt
        image_set_file = os.path.join(self._data_path, 'image_sets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        '''
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        '''

        gt_roidb = [self._load_custom_annotation(index)
                    for index in self.image_index]

        '''
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        '''

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        '''
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        '''

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            # Selective search boxes are included in the ground truth.
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
            print len(roidb)
        '''
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)
        '''

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        '''
        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['all_boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)
        '''

        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(self._data_path, 'regions', self.image_index[i] + '.mat')
            assert os.path.exists(filename), \
                'Selective search not found at: {}'.format(filename)
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:, (1, 0, 3, 2)] - 1))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        '''
        cache_file = os.path.join(self.cache_path,
                                  '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                                  format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        '''

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        '''
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)
        '''

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), \
            'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :] - 1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_custom_annotation(self, index):
        """
        Load image and bounding boxes info from mat files.
        """
        filename = os.path.join(self._data_path, 'annotations', index + '.mat')
        # print 'Loading: {}'.format(filename)

        assert os.path.exists(filename), \
            'Custom data not found at: {}'.format(filename)
        # input format: [ymin xmin ymax xmax gt_cls (gt_cls_overlap)]
        objs = sio.loadmat(filename)['boxes']
        num_objs = objs.shape[0]
        assert(objs.shape[1] >= 5)

        # boxes: [xmin ymin xmax ymax]
        boxes = (objs[:, (1, 0, 3, 2)] - 1).astype(np.uint16)
        gt_classes = (objs[:, 4]).astype(np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix in range(num_objs):
            cls = gt_classes[ix]
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def _write_custom_results_file_per_class(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # Example: devkit_path/results/test/comp4-44503_det_test_aeroplane.txt
        #dir = os.path.join(self._devkit_path, 'results', self.name)
        dir = os.path.join(self._devkit_path, 'results')
        if not os.path.exists(dir):
            os.makedirs(dir)

        #path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        path = os.path.join(self._devkit_path, 'results')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = path + '/det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))
        return comp_id

    def _write_custom_results_file(self, all_boxes):
        # Parameters.
        kScoreTol = 0.0

        color_image_dir = os.path.join(self._devkit_path, 'data/color')
        depth_image_dir = os.path.join(self._devkit_path, 'data/depth')

        result_dir = os.path.join(self._devkit_path, 'results', self.name)
        output_box_dir = os.path.join(result_dir, 'boxes')
        output_image_dir = os.path.join(result_dir, 'images')

        if not os.path.isdir(output_box_dir):
            os.makedirs(output_box_dir)

        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)

        for im_ind, index in enumerate(self.image_index):
            print 'Writing {} results file'.format(index)

            # Read images.
            color_im = Image.open(color_image_dir + '/' + index + '.png')
            depth_im = Image.open(depth_image_dir + '/' + index + '.png')

            # Rescale the 16bit depth image and convert the grayscale image to a rgb image.
            max_depth = depth_im.getextrema()[1]
            temp_im = Image.new('L', depth_im.size)
            temp_im.putdata(depth_im.getdata(), (255.0 / max_depth))
            depth_im = Image.merge('RGB', [temp_im, temp_im, temp_im])

            out_box_filename = output_box_dir + '/' + index + '.csv'

            with open(out_box_filename, 'wt') as f:
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue

                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue

                    color = custom_utils.random_color(cls_ind)

                    for k in xrange(dets.shape[0]):
                        # Format: (xmin, ymin, xmax, ymax, cls_ind, score, (roi_ind))
                        # (xmin, ymin, xmax, ymax): Starts from zero.
                        if dets.shape[1] > 5:
                            # @mhsung
                            # Record the original roi indices at the last column
                            f.write('{:.1f},{:.1f},{:.1f},{:.1f},{:d},{:.3f},{:d}\n'.
                                    format(dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3],\
                                        cls_ind, dets[k, 4], int(dets[k, 5])))
                        else:
                            f.write('{:.1f},{:.1f},{:.1f},{:.1f},{:d},{:.3f}\n'.
                                    format(dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3],\
                                        cls_ind, dets[k, 4]))
                        score = dets[k, 4]
                        if score >= kScoreTol:
                            text = cls + '[' + str(score) + ']'
                            custom_utils.draw_rect_image(color_im,\
                                    dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], text, color)
                            custom_utils.draw_rect_image(depth_im,\
                                    dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], text, color)

            #width, height = color_im.size
            #im = Image.new('RGB', (2 * width, height))
            #im.paste(color_im, (0, 0))
            #im.paste(depth_im, (width, 0))

            color_image_file = output_image_dir + '/' + index + '_color.png'
            color_im.save(color_image_file)
            depth_image_file = output_image_dir + '/' + index + '_depth.png'
            depth_im.save(depth_image_file)


    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
            .format(self._devkit_path, comp_id,
                    self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_custom_results_file(all_boxes)
        comp_id = self._write_custom_results_file_per_class(all_boxes)
        # self._do_matlab_eval(comp_id, output_dir)
        return

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = datasets.custom('train', '')
    res = d.roidb
    from IPython import embed;

    embed()
