import torch
import torchvision

from util import transforms as ctransforms
from util.common import plot_box


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