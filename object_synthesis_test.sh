cp ~/home/data/object_synthesis/labels.txt ~/home/data/test_nyud2_custom/

./tools/test_net.py --gpu 1 --def models/object_synthesis_VGG16/test.prototxt \
  --net output/default/train/object_vgg16_fast_rcnn_iter_40000.caffemodel --imdb custom_test

mv ~/home/data/test_nyud2_custom/results ~/home/data/test_nyud2_custom/object_results
