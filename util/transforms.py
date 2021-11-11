import torch
import torchvision


class resize:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.ar = H / W # aspect ratio

    def __call__(self, img, box):
        '''
        To maintain aspect ratio when resizing, we will pad the necessary sides then resize
        We will also recalculate the bounding box label
        '''

        xmin, ymin, xmax, ymax = tuple(box)
        img_c, img_h, img_w = img.shape
        ar = img_h / img_w

        if ar > self.ar:
            # Need to pad width
            width_pad = img_h / self.ar - img_w
            half_width = int(round(width_pad / 2))
            padding = (half_width, half_width, 0, 0, 0, 0)
            img = torch.nn.functional.pad(img, padding)

            # Include padding in label
            xmin += half_width
            xmax += half_width
            img_w += half_width * 2
        elif ar < self.ar:
            # Need to pad height
            height_pad = self.ar * img_w - img_h
            half_height = int(round(height_pad / 2))
            padding = (0, 0, half_height, half_height, 0, 0)
            img = torch.nn.functional.pad(img, padding)

            # Include padding in label
            ymin += half_height
            ymax += half_height
            img_h += half_height * 2

        # Resize img
        img = torchvision.transforms.functional.resize(img, (self.H, self.W))

        # Resize labels
        xmin, xmax = self.W * (xmin / img_w), self.W * (xmax / img_w)
        ymin, ymax = self.H * (ymin / img_h), self.H * (ymax / img_h)

        return img, torch.tensor([xmin, ymin, xmax, ymax])
