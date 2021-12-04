import os
import torch
import matplotlib.pyplot as plt

from module.models import FeatureExtractor
from tqdm import tqdm


class Trainer:
    def __init__(self, train_dataloader, valid_dataloader, test_dataloader, epochs=20, **kwargs):

        # Defining constants
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
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

    def plot_curve(self):
        x = [_ for _ in range(20)]
        train_loss = [1.953, 1.3882, 1.1474, 1.03599, 0.90604, 0.6764, 0.613966, 0.5789, 0.5383, 0.51047, 0.4753,
                      0.47789, 0.464825,0.460503, 0.467663, 0.444439, 0.45601359, 0.459355, 0.4650490, 0.4553926]
        val_loss = [1.53853261, 1.26586,1.1259732,1.072498, 1.043067812919,0.92924988,0.93180048,0.8840809,0.9327643,
                    0.91877877, 0.9169676, 0.916742, 0.90229851, 0.89235621, 0.91823112, 0.920982301, 0.9059537053,
                    0.9048750,0.93360078,0.917428672]
        val_acc = [0.5493562, 0.6299977, 0.6634289, 0.6873729, 0.691664784,  0.728258414, 0.72215947,  0.741811610,
                   0.729387847,0.733905,0.73910097,0.7377456,0.73797153,0.74700700,0.7370679,0.74045629,0.7433928,
                   0.74361870,0.73616444,0.738875084]
        plt.figure()
        plt.plot(x, train_loss, label='train loss')
        # plt.plot(x, val_loss, label='val loss')
        plt.legend()
        plt.show()
        # plt.savefig(f'{self.save_dir}/loss-curve.png')

        # plt.plot(val_acc, label='val accuracy')

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

    def test(self):
        import numpy as np
        from util.voc_dataset import VOCDataset
        total = 0
        correct = 0
        self.extractor.eval()
        for image, label in tqdm(self.test_dataloader):
            image, label = image.to(self.device), label.squeeze(1).to(self.device)
            out = self.extractor(image)
            predicted_label = torch.argmax(out, dim=1)
            total += label.size(0)
            correct += (predicted_label.float() == label.float()).sum().item()
            predicted_label = VOCDataset.get_idx2cls()[predicted_label[0].item()]
            print(predicted_label)
            temp = image[0, :, :, :].cpu().permute(1,2,0)
            temp = np.ascontiguousarray(temp, dtype=np.uint8)
            plt.figure()
            plt.imshow(temp)
            plt.show()

        print("Test accuracy: ", correct / total)

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

