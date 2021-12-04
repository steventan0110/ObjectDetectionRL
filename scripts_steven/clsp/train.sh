ROOT=${HOME}/ObjectDetectionRL
EXPORT_FOLDER=/export/b08/wtan/rl
python ${ROOT}/main.py \
  --mode train \
  --data_dir ${EXPORT_FOLDER}/dataset \
  --save_dir ${EXPORT_FOLDER}/checkpoints/
