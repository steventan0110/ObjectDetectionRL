import os
import torch

import matplotlib.pyplot as plt
from module.models import DQN, FeatureExtractor
from tqdm import tqdm


class Trainer:
    def __init__(self, train_dataloader, valid_dataloader, epochs=20, **kwargs):

        # Defining constants
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.height = kwargs['height']
        self.width = kwargs['width']
        self.epochs = epochs
        self.save_interval = kwargs['save_interval']
        self.save_dir = kwargs['save_dir']

        # Defining hyperparameters
        self.batch_size = kwargs['batch_size']
        self.lr = kwargs['learning_rate']
        self.target_update = kwargs['target_update']

        # Initializing models
        self.extractor = FeatureExtractor(kwargs['image_extractor'], freeze=False).to(self.device)
        self.extractor.train()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.extractor.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.save_interval, gamma=0.1)

    def plot_curve(self, train_loss, val_loss, val_acc):
        plt.figure()
        plt.plot(train_loss, label='train loss')
        plt.plot(val_loss, label='val loss')
        plt.plot(val_acc, label='val accuracy')
        plt.legend()
        plt.savefig(f'{self.save_dir}/loss-curve.png')


    def train(self):
        best_val = float('inf')
        train_loss_all = []
        val_loss_all = []
        val_acc_all = []
        for epoch in range(self.epochs):
            batch_loss = []
            self.extractor.train()
            for image, label in tqdm(self.train_dataloader):
                self.optimizer.zero_grad()
                image, label = image.to(self.device), label.squeeze(1).to(self.device)
                out = self.extractor(image)
                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss)

            train_loss = sum(batch_loss) / len(self.train_dataloader)
            train_loss_all.append(train_loss)
            with torch.no_grad():
                self.extractor.eval()
                valid_loss, val_acc = self.validate()
                val_loss_all.append(valid_loss)
                val_acc_all.append(val_acc)

            print(f'Epoch {epoch+1}: Train loss: {train_loss}, Valid loss: {valid_loss}, Valid Acc: {val_acc}')
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, self.save_dir)

            if best_val > valid_loss:
                best_val = valid_loss
                self.save_checkpoint("best", self.save_dir)
            self.scheduler.step()

        self.plot_curve(train_loss_all, val_loss_all, val_acc_all)


    def validate(self):
        valid_batch_loss = []
        total = 0
        correct = 0
        for image, label in tqdm(self.valid_dataloader):
            image, label = image.to(self.device), label.squeeze(1).to(self.device)
            out = self.extractor(image)
            loss = self.criterion(out, label)
            valid_batch_loss.append(loss)

            predicted_label = torch.argmax(out, dim=1)
            total += label.size(0)
            correct += (predicted_label.float() == label.float()).sum().item()

        return sum(valid_batch_loss) / len(self.valid_dataloader), correct/total


    def save_checkpoint(self, epoch, dir):
        save_path = os.path.join(dir, 'checkpoint_vgg_' + str(epoch) + '.pt')
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save({
            'extractor': self.extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def load_checkpoint(self, param_path):
        checkpoint = torch.load(param_path, map_location=self.device)
        self.extractor.load_state_dict(checkpoint['extractor'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

