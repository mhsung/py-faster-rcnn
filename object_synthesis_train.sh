./tools/train_net.py --gpu 1 --solver models/object_synthesis_VGG16/solver.prototxt \
  --weights models/object_synthesis_VGG16/transplanted.caffemodel \
  --imdb object_synthesis_train 2>&1 | tee ./logs/object_synthesis_train.log
