import cv2
import json
import matplotlib.pyplot as plt

from util import split_data as sd


def test_process_xml_file():
    label_path = '/home/shuhao/Downloads/dataset/train-dev/2007/labels/000005.xml'
    img_path = '/home/shuhao/Downloads/dataset/train-dev/2007/images/000005.jpg'
    labels = sd.process_xml_file(label_path)
    image = cv2.imread(img_path)
    blue = (255, 0, 0)

    for label in labels:
        xmin, ymin, xmax, ymax = label['box']
        name = label['name']
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), blue, 2)
        cv2.putText(image, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.9, blue, 2)

    plt.figure()
    plt.imshow(image)
    plt.show()


def read_json_labels():
    json_path = '/home/shuhao/Downloads/dataset/train/labels.json'
    with open(json_path) as f:
        data = json.load(f)
        print(data['000017.jpg'])


def check_class_balance():
    def gen_hist(json_path):
        with open(json_path) as f:
            counts = dict()
            data = json.load(f)
            for img_name, objs in data.items():
                for obj in objs:
                    name = obj['name']
                    if name in counts:
                        counts[name] = counts[name] + 1
                    else:
                        counts[name] = 1
            return counts

    def plot_hist(dict_data, title):
        plt.figure()
        dict_items = list(dict_data.items())
        dict_items.sort(key=lambda item:item[0]) # sort by class name
        plt.bar(*zip(*dict_items))
        plt.title(title)
        plt.xlabel('Classes')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        plt.show()

    train_json = '/home/shuhao/Downloads/dataset/train/labels.json'
    val_json = '/home/shuhao/Downloads/dataset/val/labels.json'
    test_json = '/home/shuhao/Downloads/dataset/test/labels.json'

    train_data = gen_hist(train_json)
    val_data = gen_hist(val_json)
    test_data = gen_hist(test_json)

    plot_hist(train_data, 'Train Data')
    plot_hist(val_data, 'Val Data')
    plot_hist(test_data, 'Test Data')


def test_sort_labels():
    labels_folder = '/home/jun/Downloads/dataset/test'
    breakpoint() # Good point to start debugging this method
    sd.sort_labels(labels_folder)


def run_all():
    test_process_xml_file()
    read_json_labels()
    check_class_balance()
    test_sort_labels()