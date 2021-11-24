import torch
import torchvision


class resize:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.ar = H / W # aspect ratio

    def __call__(self, img, boxes):
        '''
        To maintain aspect ratio when resizing, we will pad the necessary sides then resize
        We will also recalculate the bounding box label
        '''

        boxes = torch.clone(boxes) # Avoid changing the input boxes tensor
        img_c, img_h, img_w = img.shape
        ar = img_h / img_w

        if ar > self.ar:
            # Need to pad width
            width_pad = img_h / self.ar - img_w
            half_width = int(round(width_pad / 2))
            padding = (half_width, half_width, 0, 0, 0, 0)
            img = torch.nn.functional.pad(img, padding)
            img_w += half_width * 2

            # Include padding in label
            for b_i, box in enumerate(boxes):
                # box = tensor([xmin, ymin, xmax, ymax])
                box[0] += half_width
                box[2] += half_width
        elif ar < self.ar:
            # Need to pad height
            height_pad = self.ar * img_w - img_h
            half_height = int(round(height_pad / 2))
            padding = (0, 0, half_height, half_height, 0, 0)
            img = torch.nn.functional.pad(img, padding)
            img_h += half_height * 2

            # Include padding in label
            for b_i, box in enumerate(boxes):
                # box = tensor([xmin, ymin, xmax, ymax])
                box[1] += half_height
                box[3] += half_height

        # Resize img
        img = torchvision.transforms.functional.resize(img, (self.H, self.W))

        # Resize labels
        for b_i, box in enumerate(boxes):
            # xmin
            box[0] = self.W * (box[0] / img_w)
            # ymin
            box[1] = self.H * (box[1] / img_h)
            # xmax
            box[2] = self.W * (box[2] / img_w)
            # ymax
            box[3] = self.H * (box[3] / img_h)

        return img, boxes
