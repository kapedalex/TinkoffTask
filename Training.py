import Parser
import Pipeline
import torch
import torch.nn as nn
import torch.optim
import numpy as np

from Model import Model
from ModelWorker import ModelWorker
from Resources.Constants import Constants


class Train(ModelWorker):

    def __init__(self):
        super().__init__()
        self.args = Parser.get_train_args()
        self.sequence, self.char_to_idx, self.idx_to_char = Pipeline.get_pipelined_data()
        self.size = len(self.idx_to_char)
        self.model = Model(input_size=len(self.idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=5,
            verbose=True,
            factor=0.5
        )

        self.train_losses = []

    def get_batch(self, sequences):
        trains = []
        targets = []
        for _ in range(self.args.batch):
            batch_start = np.random.randint(0, len(sequences) - self.args.seq)
            chunk = sequences[batch_start: batch_start + self.args.seq]
            train = torch.LongTensor(chunk[:-1]).view(-1, 1)
            target = torch.LongTensor(chunk[1:]).view(-1, 1)
            trains.append(train)
            targets.append(target)
        return torch.stack(trains, dim=0), torch.stack(targets, dim=0)

    def save(self):
        torch.save(self.model.state_dict(), Constants.SAVE_PATH_MODEL)
        torch.save(self.optimizer.state_dict(), Constants.SAVE_PATH_OPTIMIZER)
        torch.save(self.train_losses, Constants.SAVE_PATH_LOSS)

    def train(self):
        loss_avg = []
        for epoch in range(self.args.epochs):
            print('training...')
            self.model.train()
            train_losses = []
            train, target = self.get_batch(self.sequence)
            train = train.permute(1, 0, 2).to(self.device)
            target = target.permute(1, 0, 2).to(self.device)
            hidden = self.model.init_hidden(self.device, self.args.batch)

            output, hidden = self.model(train, hidden)
            loss = self.criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

            loss.backward()
            train_losses.append(loss.item())
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_avg.append(loss.item())

            if len(loss_avg) >= 50:
                mean_loss = np.mean(loss_avg)
                print(f'Loss: {mean_loss}')
                self.scheduler.step(mean_loss)
                loss_avg = []
                self.model.eval()
                predicted_text = self.evaluate(self.model, self.char_to_idx, self.idx_to_char,
                                               start_text=self.args.prefix, prediction_len=self.args.length)
                print(predicted_text)
                self.save()

            self.save()
