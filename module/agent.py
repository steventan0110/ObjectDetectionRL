import os
import torch
from module.models import DQN, FeatureExtractor

class Agent:
	def __init__(self, train_dataloader, valid_dataloader,
	             alpha=0.2, threshold=0.6, eta=3,
	             episodes=100, max_steps_per_episode=20,
	             **kwargs):
		self.args = kwargs
		self.lr = kwargs['learning_rate']
		self.train_dataloader = train_dataloader
		self.valid_dataloader = valid_dataloader
		self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
		self.height = kwargs['height']
		self.width = kwargs['width']
		self.episodes = episodes
		self.max_steps_per_episode = max_steps_per_episode
		self.extractor = FeatureExtractor(kwargs['image_extractor']).to(self.device)
		self.extractor.eval() # do not update the pretrained CNN
		print(self.extractor)
		self.alpha = alpha
		self.threshold = threshold
		self.eta = eta
		self.epsilon = 0.2
		self.batch_size = kwargs['batch_size']
		rl_algo = kwargs['rl_algo']
		if rl_algo == 'DQN':
			self.policy_net= DQN().to(self.device)
			self.target_net = DQN() # DQN use two policy network
			self.target_net.load_state_dict(self.policy_net.state_dict())
			self.target_net.eval()
			self.target_update = 20 # TODO: hardcode for now
		print(self.policy_net)

		# self.criterion = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.args['learning_rate'])
		self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
			self.optimizer,
			max_lr=self.args['learning_rate'],
			epochs=self.args['max_epoch'],
			steps_per_epoch=len(self.train_dataloader))


	def get_state(self, image):
		image_feature = self.extractor(image) # (bz, 512, 16, 16) if image is not resized
		bz = image_feature.size(0)
		image_feature = image_feature.view(bz, -1)
		history = self.actions_history.view(bz, -1).type(torch.FloatTensor).to(self.device)
		state = torch.cat((image_feature, history), 1)
		return state

	def get_action(self, state):
		p = torch.rand(1).item()
		if p > self.epsilon:
			pred = self.policy_net(state)
			_, best_action = torch.max(pred, 1)
			return best_action
		else:
			return torch.randint(0, 9, (self.batch_size, ))

	def update_box(self, actions, prev_boxs, done):
		# we have a batch so each box is updated differently
		# need to take a mask that tracks done
		new_box = prev_boxs.new_full((self.batch_size, 4), 0)
		# TODO: I can't think of a way to parallelize here, since it's not computational heavy, I think use for loop
		#  here is okay
		for i in range(self.batch_size):
			action = int(actions[i].item())
			if action == 8:
				done[i] = True

		return new_box, done

	def train(self):
		xmin = 0.0
		xmax = 224.0
		ymin = 0.0
		ymax = 224.0

		for i_episode in range(self.episodes):
			for idx, (images, boxes, classes) in enumerate(self.train_dataloader):
				images, boxes, classes = images.to(self.device), boxes.to(self.device), classes.to(self.device)
				# history of past 9 actions, each consists of a one hot vec
				self.actions_history = torch.ones((self.batch_size,9,9)).to(self.device)

				state = self.get_state(images) # bz x (81 + image feature size)
				original_coordinates = torch.tensor([xmin, xmax, ymin, ymax]).repeat(self.batch_size, 1).to(self.device)
				prev_box = original_coordinates
				cur_box = original_coordinates
				t = 0
				done = prev_box.new_full((self.batch_size, 1), 0).bool().squeeze(1) # boolean mask
				while not torch.all(done):
					action = self.get_action(state)
					# TODO: update the bounding box
					cur_box, done = self.update_box(action, prev_box, done)
					print(cur_box, done)
					raise Exception
					# TODO: update the observed region (change image region)

					#TODO: compute reward

					# TOOD: backprop to update policy network
					prev_box = cur_box
					t += 1
					if t == self.max_steps_per_episode:
						break

			if i_episode % self.target_update == 0:
				self.target_net.load_state_dict(self.policy_net.state_dict())


	def save_checkpoint(self, epoch, loss, val_loss, dir):
		save_path = os.path.join(dir, 'checkpoint' + str(epoch) + '.pt')
		torch.save({
			'epoch': epoch,
			'loss': loss,
			'val_loss': val_loss,
			'policy_network': self.policy_net.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
		}, save_path)

	def load_checkpoint(self, dir):
		checkpoint = torch.load(dir, map_location=self.device)
		self.policy_net.load_state_dict(checkpoint['policy_network'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


