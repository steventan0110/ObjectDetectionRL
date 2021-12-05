# Best LR found: 0.00001

CLS=$1

DATA=/raid/home/slai16/ObjectDetectionRL/dataset
ROOT=/raid/home/slai16/ObjectDetectionRL/${CLS}

for model in "DQN" "DQN" ""; do
  echo "EVALUATING MODEL TRANSFER LEARNED FROM FAT LANDMARK $counter"

  FILES="$TASK_FOLDER/val_image_paths.txt $TASK_FOLDER/val_landmark_paths.txt"
  PARAM_FILE="$PARAM_FOLDER/transfer_from_set_1_fat_$counter/dev/model-100000"
  LOG_FOLDER="$VAL_LOGS/transfer_from_set_1_fat_$counter"

#    echo "$PARAM_FILE"
#    echo "$FILES"
#    echo "$LOG_FOLDER"

  python DQN.py \
    --task eval  \
    --gpu 0 \
    --load $PARAM_FILE \
    --files $FILES \
    --logDir $LOG_FOLDER \
    --saveGif \
    --agents 1
done



mkdir ${ROOT}
mkdir ${ROOT}/checkpoints
mkdir ${ROOT}/stats

python main.py \
  --mode train \
  --batch_size 100 \
  --data_dir ${DATA}\
  --save_dir ${ROOT}/checkpoints/ \
  --stats_dir ${ROOT}/stats/aeroplane.json \
  --learning_rate 0.00001 \
  --epochs 25 \
  --rl_algo DQN \
  --cls ${CLS}