ROOT=/home/steven/Code/GITHUB/ObjectDetectionRL

python ${ROOT}/main.py \
  --mode pretrain \
  -lr 1e-4 \
  --load-path-cnn ${ROOT}/checkpoints/checkpoint_vgg_best.pt \
  --data_dir ${ROOT}/dataset \
  --save_dir ${ROOT}/checkpoints/
