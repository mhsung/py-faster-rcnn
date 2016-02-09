./tools/train_net.py --gpu 0 --solver models/part_synthesis_VGG16/solver.prototxt \
  --weights models/part_synthesis_VGG16/transplanted.caffemodel \
  --imdb part_synthesis_train 2>&1 | tee ./logs/part_synthesis_train.log
