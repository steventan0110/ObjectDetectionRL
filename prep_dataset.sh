ROOT=/home/steven/Code/GITHUB/ObjectDetectionRL
mkdir -p ${ROOT}/dataset/train
mkdir -p ${ROOT}/dataset/dev
mkdir -p ${ROOT}/dataset/test
mkdir -p ${ROOT}/dataset/train-dev

extract_file() {
    # retrieve the data as label, ignore other files from VOC dataset
    year=$1
    des_dir=$2
    if [ -e ${des_dir} ]; then
        # avoid duplicate data
        rm -rf ${des_dir}
    fi
    mkdir -p $des_dir/images
    mkdir -p $des_dir/labels
    cp ${ROOT}/VOCdevkit/VOC${year}/JPEGImages/* ${des_dir}/images
    cp ${ROOT}/VOCdevkit/VOC${year}/Annotations/* ${des_dir}/labels
    #rm -rf ${ROOT}/VOCdevkit   
}

# 2012 train val dataset
wget pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar -xvf ${ROOT}/VOCtrainval_11-May-2012.tar
extract_file 2012 ${ROOT}/dataset/train-dev/2012
rm ${ROOT}/VOCtrainval_11-May-2012.tar

# 2007 train val dataset
wget pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
tar -xvf ${ROOT}/VOCtrainval_06-Nov-2007.tar
extract_file 2007 ${ROOT}/dataset/train-dev/2007
rm ${ROOT}/VOCtrainval_06-Nov-2007.tar

# 2007 test file
wget pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xvf ${ROOT}/VOCtest_06-Nov-2007.tar
extract_file 2007 ${ROOT}/dataset/test/2007
rm ${ROOT}/VOCtest_06-Nov-2007.tar

# separate file from train-dev into train and dev
${ROOT}/dataset/train-dev/