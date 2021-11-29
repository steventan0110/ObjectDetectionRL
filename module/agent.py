import os
import torch
from module.models import DQN, FeatureExtractor
from module.memory import ReplayMemory
from torch.autograd import Variable


class Agent:

    def __init__(self, train_dataloader, valid_dataloader,
                 alpha=0.2, threshold=0.6, eta=3,
                 episodes=100, max_steps_per_episode=20,
                 **kwargs):
        self.num_args = 9
        self.args = kwargs
        self.gamma = 0.7
        self.lr = kwargs['learning_rate']
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
        self.height = kwargs['height']
        self.width = kwargs['width']
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.extractor = FeatureExtractor(kwargs['image_extractor']).to(self.device)
        self.extractor.eval()  # do not update the pretrained CNN
        print(self.extractor)
        self.alpha = alpha
        self.threshold = threshold
        self.eta = eta
        self.epsilon = 0.2
        self.batch_size = kwargs['batch_size']
        rl_algo = kwargs['rl_algo']
        if rl_algo == 'DQN':
            self.policy_net = DQN().to(self.device)
            self.target_net = DQN().to(self.device)  # DQN use two policy network
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.target_update = 20  # TODO: hardcode for now
        print(self.policy_net)
        self.memory = ReplayMemory(1000) # capacity = 1000
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.args['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args['learning_rate'],
            epochs=self.args['max_epoch'],
            steps_per_epoch=len(self.train_dataloader))

    def get_state(self, image):
        image_feature = self.extractor(image)  # (bz, 512, 16, 16) if image is not resized
        image_feature = image_feature.reshape(self.batch_size, -1)
        history = self.actions_history.reshape(self.batch_size, -1).type(torch.FloatTensor).to(self.device)
        state = torch.cat((image_feature, history), 1)
        return state

    def get_action(self, state):
        p = torch.rand(1).item()
        if p > self.epsilon:
            pred = self.policy_net(state)
            _, best_action = torch.max(pred, 1)
            return best_action
        else:
            # TODO
            # This should be random positive action rather than just random action
            # There are no situations where a random action that reduces reward is optimal
            return torch.randint(0, 9, (self.batch_size,))

    def update_action_history(self, action):
        # Update action history
        # Always rolls the history so that the earliest action is no longer tracked
        # Note that action history is initialize to tensor of ones; the ones will be
        # the first to get rolled off

        # self.action_history has shape (number of actions, size of action space)
        self.actions_history = torch.roll(self.actions_history, 1, 0)
        action_one_hot = torch.zeros(self.num_args)
        action_one_hot[action] = 1
        self.actions_history[0] = action_one_hot

    def update_box(self, action, prev_box):
        x_min, x_max, y_min, y_max = prev_box[0], prev_box[1], prev_box[2], prev_box[3]
        alpha_h = self.alpha * (y_max - y_min)
        alpha_w = self.alpha * (x_max - x_min)

        if action == 0:  # left
            x_min -= alpha_w
            x_max -= alpha_w
        elif action == 1:  # right
            x_min += alpha_w
            x_max += alpha_w
        elif action == 2:  # up
            y_min -= alpha_h
            y_max -= alpha_h
        elif action == 3:  # down
            y_min += alpha_h
            y_max += alpha_h
        elif action == 4:  # bigger
            x_min -= alpha_w
            x_max += alpha_w
            y_min -= alpha_h
            y_max += alpha_h
        elif action == 5:  # smaller
            x_min += alpha_w
            x_max -= alpha_w
            y_min += alpha_h
            y_max -= alpha_h
        elif action == 6:  # shorter
            y_min += alpha_h
            y_max -= alpha_h
        elif action == 7:  # narrower
            x_min += alpha_w
            x_max -= alpha_w
        elif action == 8:
            # New no box coordinates and done is True
            return None, True

        x_min = min(max(x_min, 0), self.width)
        x_max = min(max(x_max, 0), self.width)
        y_min = min(max(y_min, 0), self.height)
        y_max = min(max(y_max, 0), self.height)

        new_box = [x_min, x_max, y_min, y_max]
        return new_box, False

    def update_observed_region_old(self, image, box):
        import torchvision.transforms as T
        x_min, x_max, y_min, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # resize back to the same height and width
        resize = T.Compose([
            T.Resize(self.width),
            T.CenterCrop(self.width)
        ])
        # TODO: is this proper? especially if the action is fatter/thinner, the image would be stretched
        # sometimes get division by 0 error
        image = resize(image[:, x_min:x_max, y_min:y_max])
        return image

    def update_observed_region(self, image, box):
        from util.transforms import resize
        x_min, x_max, y_min, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_roi = image[:, x_min:x_max, y_min:y_max]
        transform = resize(self.height, self.width)
        image, _box = transform(img_roi, box)
        return image

    def find_closest_box(self, box, gt_boxes):
        max_area = -float('inf')
        best_box = None
        for gt_box in gt_boxes:
            gt_box = gt_box.numpy()
            area = self.compute_IOU(box, gt_box)
            if area > max_area:
                max_area = area
                best_box = gt_box
        return max_area, best_box

    def compute_IOU(self, box, ground_truth):
        x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
        x_min_gold, x_max_gold, y_min_gold, y_max_gold = ground_truth[0], ground_truth[1], \
                                                         ground_truth[2], ground_truth[3]

        box_area = (x_max - x_min) * (y_max - y_min)
        ground_truth_area = (x_max_gold - x_min_gold) * (y_max_gold - y_min_gold)
        inner_area = (min(x_max, x_max_gold) - max(x_min, x_min_gold)) * (
                    min(y_max, y_max_gold) - max(y_min, y_min_gold))
        inner_area = max(inner_area, 0)
        union_area = box_area + ground_truth_area - inner_area
        return inner_area / union_area

    def compute_reward(self, prev_box, cur_box, ground_truth_boxes):
        IOU_prev, _best_box = self.find_closest_box(prev_box, ground_truth_boxes)
        IOU_cur, _best_box = self.find_closest_box(cur_box, ground_truth_boxes)
        if IOU_cur > IOU_prev:
            return 1
        return 0

    def train(self):
        xmin = 0.0
        xmax = self.width
        ymin = 0.0
        ymax = self.height

        for i_episode in range(self.episodes):
            for idx, (image, boxes) in enumerate(self.train_dataloader):
                # BATCHSIZE=1 is fixed, ignore this index
                image, boxes = image.squeeze(0).to(self.device), boxes.squeeze(0).to(self.device)

                # Action history is initialized to all ones when episode first starts
                # This allows us to have a fix number of states when passing the features to the Q function
                # Real actions will be represented with a one hot vector
                # TODO Can also try using all zeros.
                self.actions_history = torch.ones((9, 9), device=self.device)

                original_coordinates = [xmin, xmax, ymin, ymax]
                prev_state = self.get_state(image)  # bz x (81 + image feature size)
                prev_box = original_coordinates
                prev_image = image
                done = False

                t = 0
                while not done and t < self.max_steps_per_episode:
                    action = self.get_action(prev_state).item()
                    self.update_action_history(action)
                    cur_box, done = self.update_box(action, prev_box)
                    if done:
                        # Termination reward is different than reward from other actions
                        IOU, _best_box = self.find_closest_box(prev_box, boxes)
                        reward = self.eta if IOU >= self.threshold else -1 * self.eta
                        self.memory.push(prev_state, action, None, reward)
                    else:
                        try:
                            cur_image = self.update_observed_region(prev_image, cur_box)
                        except ValueError:
                            # Due to stochastic nature of bounding box values from model,
                            # there are cases when exceptions occur
                            break

                        next_state = self.get_state(cur_image)
                        reward = self.compute_reward(prev_box, cur_box, boxes)
                        self.memory.push(prev_state, action, next_state, reward)

                        # Tracked object update for next loop
                        prev_box = cur_box
                        prev_image = cur_image
                        prev_state = next_state

                    self.train_policy_net(4)
                    t += 1

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_policy_net(self, BATCH_SIZE=None):
        # Optimize the model using backprop

        if BATCH_SIZE is None:
            BATCH_SIZE = self.batch_size
        if len(self.memory) < BATCH_SIZE:
            return

        samples_tuple = self.memory.sample(BATCH_SIZE)
        rewards = torch.zeros((BATCH_SIZE, 1), device=self.device)
        actions = torch.zeros((BATCH_SIZE, 1), dtype=torch.int64, device=self.device)
        final_state_mask = torch.zeros(BATCH_SIZE)
        cur_state_list, next_state_list = [], []
        for i in range(BATCH_SIZE):
            cur_state, action, next_state, reward = samples_tuple[i]
            cur_state_list.append(cur_state)
            rewards[i] = reward
            actions[i] = action
            if next_state is not None:
                next_state_list.append(next_state)
            else:
                final_state_mask[i] = 1
                empty_state = torch.zeros((1, cur_state.size(1)))
                next_state_list.append(empty_state)
        cur_state_tensor = Variable(torch.stack(cur_state_list, dim=0).squeeze().to(self.device))
        next_state_tensor = Variable(torch.stack(next_state_list, dim=0).squeeze().to(self.device))

        # Calculating predicted q values
        Q = self.policy_net(cur_state_tensor).gather(1, actions)

        # Calculating target q values
        next_state_action_value = self.target_net(next_state_tensor)
        max_next_state_action_value, _ = torch.max(next_state_action_value, dim=1)
        masked_next_value = max_next_state_action_value * (1 - final_state_mask)
        expected_Q = rewards + self.gamma * masked_next_value

        loss = self.criterion(Q, expected_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
