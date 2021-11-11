import cv2
import torch
import torchvision
import matplotlib.pyplot as plt

from util import transforms as ctransforms


def test_resize():
    def plot_box(img, box, title):
        start = int(box[0]), int(box[1])
        end = int(box[2]), int(box[3])
        img = img.numpy().transpose(1, 2, 0)
        img = img.copy()  # To make array contiguous; https://stackoverflow.com/a/31316516
        img = cv2.rectangle(img, start, end, (255, 0, 0), 2)
        plt.figure()
        plt.imshow(img)
        plt.title(title)
        plt.show()

    img_path = '/home/shuhao/Downloads/dataset/test/images/000003.jpg'
    H, W = 256, 256
    resize = ctransforms.resize(H, W)

    img = torchvision.io.read_image(img_path)
    label = torch.tensor([123, 155, 215, 195]) # Class sofa
    plot_box(img, label, title='Before')

    img, label = resize(img, label)
    plot_box(img, label, 'After')


def run_all():
    test_resize()