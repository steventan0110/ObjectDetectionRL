import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from util import transforms as ctransforms


def plot_box(img, box, title='Img w Box'):
    start = int(box[0]), int(box[1])
    end = int(box[2]), int(box[3])
    img = img.numpy().transpose(1, 2, 0)
    img = np.ascontiguousarray(img, dtype=np.uint8)  # Necessary to make array contiguous and be in the right dtpye; https://stackoverflow.com/a/31316516
    img = cv2.rectangle(img, start, end, (255, 0, 0), 2)
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()


def test_resize():
    img_path = '/home/jun/Downloads/dataset/test/images/000003.jpg'
    H, W = 256, 256
    resize = ctransforms.resize(H, W)

    img = torchvision.io.read_image(img_path)
    boxes = torch.tensor([[123, 155, 215, 195]]) # Class sofa
    plot_box(img, boxes[0], title='Before')

    img, boxes = resize(img, boxes)
    plot_box(img, boxes[0], 'After')


def run_all():
    test_resize()