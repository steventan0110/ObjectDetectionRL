import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_tensor(img):
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = np.ascontiguousarray(img, dtype=np.uint8)  # Necessary to make array contiguous and be in the right dtpye; https://stackoverflow.com/a/31316516
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_box(img, box, color=(255, 0, 0)):
    start = int(box[0]), int(box[1])
    end = int(box[2]), int(box[3])
    if torch.is_tensor(img):
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray(img, dtype=np.uint8)  # Necessary to make array contiguous and be in the right dtpye; https://stackoverflow.com/a/31316516
    img = cv2.rectangle(img, start, end, color, 2)
    return img


def plot_box(img, box, title='Img w Box'):
    img = draw_box(img, box)
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()