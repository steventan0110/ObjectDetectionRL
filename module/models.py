import torch
import torch.nn as nn
import torchvision


class FeatureExtractor(nn.Module):
	def __init__(self, network='vgg16', freeze=True):
		super(FeatureExtractor, self).__init__()
		if network == 'vgg16':
			model = torchvision.models.vgg16(pretrained=True)
		elif network == 'resnet50':
			model = torchvision.models.resnet50(pretrained=True)
		else:
			model = torchvision.models.alexnet(pretrained=True)

		self.features = list(model.children())[0]
		self.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
		num_cls = 20
		# replicate VGG structure
		self.new_classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(512*8*8, 4096),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Linear(4096, num_cls)
		)


		if freeze:
			self.finetune = False
			model.eval()  # to not do dropout
			for param in self.features.parameters():
				param.requires_grad = False
			for param in self.classifier.parameters():
				param.requires_grad = False
		else:
			self.finetune = True
			model.train()


	def forward(self, x):
		if not self.finetune:
			x = self.features(x)
			return x
		else:
			x = self.features(x)
			bz = x.size(0)
			x = x.view(bz, -1)
			return self.new_classifier(x)

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.classifier = nn.Sequential(
			# TODO: this dimensino should be auto computed from height and width instead of hardcode
			nn.Linear(in_features=512*8*8+81, out_features=1024),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(in_features=1024, out_features=1024),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(in_features=1024, out_features=9)
		)

	def forward(self, x):
		return self.classifier(x)


class DuelingDQN(nn.Module):
	def __init__(self, num_actions=9):
		super(DuelingDQN, self).__init__()
		self.num_actions = num_actions

		# Must output advantage for each action given input state
		self.advantage_func = nn.Sequential(
			nn.Linear(in_features=512*8*8+81, out_features=1024),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(in_features=1024, out_features=256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(in_features=256, out_features=num_actions)
		)

		# Must output scalar to represent the value of the input state
		self.value_func = nn.Sequential(
			nn.Linear(in_features=512*8*8+81, out_features=1024),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(in_features=1024, out_features=256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(in_features=256, out_features=1)
		)

	def forward(self, x):
		adv = self.advantage_func(x) # batch_size x num_actions
		adv_avg = torch.mean(adv, dim=1, keepdim=True) # batch_size x 1
		val = self.value_func(x) # batch_size x 1
		q = val + adv - adv_avg
		return q

