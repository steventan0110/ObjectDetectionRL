import torchvision
import numpy as np
import json
import torch
import os

from torch.utils.data import Dataset


'''
View test_dataset for example usage
'''


class VOCDataset(Dataset):
	def __init__(self, data_folder, label_image_transform=None, img_transform=None):
		# Be careful with transformations that change location of bounding box!
		self.label_image_transform = label_image_transform
		self.img_transform = img_transform
		self.labels, self.objects, self.cls2idx = self.parse_labels(data_folder)
		self.image_folder = os.path.join(data_folder, 'images')

	def parse_labels(self, data_path):
		json_path = os.path.join(data_path, 'labels.json')
		with open(json_path) as f:
			labels = json.load(f)
			objects = []
			obj_class = set()
			for img_name, img_objs in labels.items():
				for obj_i, obj in enumerate(img_objs):
					objects.append((img_name, obj_i))
					obj_class.add(obj['name'])

			obj_class = list(obj_class)
			obj_indices = np.arange(len(obj_class))
			cls2idx = dict(zip(obj_class, obj_indices))
			return labels, objects, cls2idx

	def __getitem__(self, idx):
		img_name, obj_i = self.objects[idx]

		# Formatting label information
		img_labels = self.labels[img_name]
		obj = img_labels[obj_i]
		box = torch.tensor(obj['box'])
		obj_class = torch.tensor(self.cls2idx[obj['name']]) # Throws error if class is unknown

		# Formatting image information
		img_path = os.path.join(self.image_folder, img_name)
		img = torchvision.io.read_image(img_path)

		if self.label_image_transform:
			for transform in self.label_image_transform:
				img, box = transform(img, box)
		if self.img_transform:
			img = self.img_transform(img)

		return img, box, obj_class

	def __len__(self):
		return len(self.objects)
