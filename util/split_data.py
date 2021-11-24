import xml.etree.ElementTree as ET
import numpy as np
import argparse
import json
import os

from util.voc_dataset import VOCDataset


OBJECT_LABEL = 'object'
OBJECT_NAME = 'name'
OBJECT_POSE = 'pose'
OBJECT_BOX = 'bndbox'
OBJECT_LOC = ('xmin', 'ymin', 'xmax', 'ymax')


def process_xml_file(path):
    def parse_bndbox(box_obj):
        box_locs = []
        for loc_key in OBJECT_LOC:
            loc_str = box_obj.find(loc_key).text.strip()
            box_locs.append(int(float(loc_str)))
        return tuple(box_locs)

    tree = ET.parse(path)
    root = tree.getroot()
    objs = root.findall(OBJECT_LABEL)
    labels = []
    for obj in objs:
        name = obj.find(OBJECT_NAME).text.strip()
        pose = obj.find(OBJECT_POSE).text.strip()
        box = parse_bndbox(obj.find(OBJECT_BOX))
        label = {
            'name': name,
            'pose': pose,
            'box': box
        }
        labels.append(label)
    return labels


def process_split(data_info, output_folder, type):
    assert type.strip().lower() in ['train', 'val', 'test']

    output_folder = os.path.join(output_folder, type)
    img_output_folder = os.path.join(output_folder, 'images')
    if not os.path.isdir(img_output_folder):
        os.makedirs(img_output_folder)

    labels_json = dict()
    for img_path, label_path in data_info:
        filename = os.path.basename(img_path)
        new_img_path = os.path.join(img_output_folder, filename)
        os.replace(img_path, new_img_path)

        labels = process_xml_file(label_path)
        labels_json[filename] = labels

    labels_path = os.path.join(output_folder, 'labels.json')
    with open(labels_path, 'w') as outfile:
        json.dump(labels_json, outfile)


def split_data(data_folders, output, train=0.7, val=0.2, seed=0):
    assert train + val <= 1.0

    data_info = []
    for data_folder in data_folders:
        img_folder = os.path.join(data_folder, 'images')
        label_folder = os.path.join(data_folder, 'labels')
        for filename in os.listdir(img_folder):
            no_ext = filename.split('.')[0].strip()
            label_path = os.path.join(label_folder, f'{no_ext}.xml')
            image_path = os.path.join(img_folder, filename)
            data = (os.path.abspath(image_path), os.path.abspath(label_path))
            data_info.append(data)
    np.random.seed(seed) # Reproducible splits
    np.random.shuffle(data_info)

    num_imgs = len(data_info)
    num_train = round(num_imgs * train)
    num_val = round(num_imgs * val)

    train_info = data_info[:num_train]
    val_info = data_info[num_train:num_train+num_val]
    test_info = data_info[num_train+num_val:]

    if train_info:
        print('Writing new training split')
        process_split(train_info, output, 'train')
        labels_folder = os.path.join(output, 'train')
        sort_labels(labels_folder)
    if val_info:
        print('Writing new validation split')
        process_split(val_info, output, 'val')
        labels_folder = os.path.join(output, 'val')
        sort_labels(labels_folder)
    if test_info:
        print('Writing new test split')
        process_split(test_info, output, 'test')
        labels_folder = os.path.join(output, 'test')
        sort_labels(labels_folder)


def sort_labels(folder):
    '''
    Given the labels for this dataset, we need to sort them by class to train one agent at a time.
    The output data will have the following structure:

    data = {
        class_name: {
            filename: [
                bndbox1,
                bndbox2, ...
            ], ...
        }, ...
    }
    '''

    json_path = os.path.join(folder, 'labels.json')
    with open(json_path) as f:
        all_labels = json.load(f)

    data = dict()
    for cls in VOCDataset.get_classes():
        data[cls] = {}

    for filename, lis_of_objs in all_labels.items():
        for obj in lis_of_objs:
            cls, pose, box = obj['name'], obj['pose'], obj['box']

            if filename not in data[cls]:
                data[cls][filename] = []

            # Not using pose information for this project
            data[cls][filename].append(box)

    json_path = os.path.join(folder, 'sorted_labels.json')
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    '''
    Example usage:
    
    Creating train and val splits from test-dev data
    python split_data.py -d /home/shuhao/Downloads/dataset/train-dev/2007 /home/shuhao/Downloads/dataset/train-dev/2012  -o /home/shuhao/Downloads/dataset -t 0.8 -v 0.2
    
    Formatting test data
    python split_data.py -d /home/shuhao/Downloads/dataset/test/2007 -o /home/shuhao/Downloads/dataset -t 0.0 -v 0.0
    '''

    parser = argparse.ArgumentParser(description='Parses Pascal object detection data')
    parser.add_argument('-d', '--data', nargs='+', required=True, help="Path to folders containing data",)
    parser.add_argument('-o', '--output', required=True, help="Path to folder where train, val, and test folders will be created")
    parser.add_argument('-t', '--train', default=0.8, type=float, help="Percent of data to be training data")
    parser.add_argument('-v', '--val', default=0.2, type=float, help="Percent of data to be validation data")
    args = parser.parse_args()

    split_data(args.data, args.output, args.train, args.val)
