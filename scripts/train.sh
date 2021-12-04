ROOT=/home/steven/Code/GITHUB/ObjectDetectionRL
mkdir -p ${ROOT}/dataset
mkdir -p ${ROOT}/checkpoints
mkdir -p ${ROOT}/stats

python ${ROOT}/main.py \
  --mode train \
  --batch_size 100 \
  --load-path-cnn ${ROOT}/checkpoints/checkpoint_vgg_best.pt \
  --data_dir ${ROOT}/dataset \
  --save_dir ${ROOT}/checkpoints/ \
  --stats_dir ${ROOT}/stats/test.json