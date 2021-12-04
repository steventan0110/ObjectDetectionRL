ROOT=/home/steven/Code/GITHUB/ObjectDetectionRL

python ${ROOT}/main.py \
  --mode test \
  --batch_size 10 \
  --load-path-cnn ${ROOT}/checkpoints/checkpoint_vgg_best.pt \
  --data_dir ${ROOT}/dataset \
  --load_path ${ROOT}/checkpoints/checkpoint_aeroplane_best.pt \
  --save_dir ${ROOT}/checkpoints/
