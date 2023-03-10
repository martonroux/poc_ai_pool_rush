import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
import torch
import numpy as np
from data_loader import get_data
import re


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(3000, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = (self.fc1(x))
        x = (self.fc2(x))
        x = (self.fc5(x))
        x = (self.fc5(x))
        x = (self.fc5(x))
        x = (self.fc3(x))
        x = (self.fc4(x))
        return torch.sigmoid(x)


class App:
    def __init__(self, lr: float, data_path: str, data_batch_size: int, epoch: int, load_path: str = None):
        if load_path is not None:
            self.model = torch.load(load_path)
        else:
            self.model = MyModel()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=lr)
        self.train_load, self.test_load = get_data(data_path, data_batch_size)
        self.batch_size = data_batch_size
        self.epoch = epoch

    def run(self):
        global_ll = []
        for epoch in range(self.epoch):
            ll = []
            for batch in self.train_load:
                self.criterion.zero_grad()
                x_batch, y_batch = batch

                y_pred = self.model.forward(x_batch.to(torch.float32))

                loss = self.criterion(y_pred, y_batch.float().reshape(len(y_pred), 1))
                ll += [loss.item()]
                loss.backward()

                self.optimizer.step()
            print(f"Epoch: {epoch + 1} / {self.epoch}, Loss: {np.mean(ll)}")
            global_ll.append(np.mean(ll))

        plt.plot(global_ll)
        plt.title('Loss over time')
        plt.savefig('loss.png')
        self.test()
        self.save_model()
        return self.model

    def test(self):
        nb_win = 0
        nb_all = 0

        for test_batch in self.test_load:
            x_batch, y_batch = test_batch

            y_pred = self.model.forward(x_batch.to(torch.float32))

            nb_all += 1

            if y_pred == y_batch:
                nb_win += 1

        print("Accuracy on test set: {}%".format(round((nb_win / nb_all) * 100), 2))

    def save_model(self):
        yes_no = input('Do you wish to save the model? [y/n]')

        while yes_no not in ['y', 'n']:
            yes_no = input('Wrong answer. Must be y/n: ')

        if yes_no == 'n':
            return
        model_name = input('Insert file name: ')
        torch.save(self.model, 'models/' + model_name + '.pt')

    def test_own_data(self, file_path: str):
        f = open(file_path, 'r')
        text = f.read().replace('\n', ' ')
        text = re.sub(r'[^a-zA-Z ]', '', text).lower()
        words_list = text.split(' ')

        new_data = pd.read_csv('data_2/empty_col.csv', sep=',')

        for word in words_list:
            if word in new_data.columns:
                new_data.loc[0, word] += 1

        x_test = torch.tensor(numpy.array(new_data))

        y_pred = self.model(x_test.to(torch.float32))

        print('The email is a: ', end='')

        if round(y_pred.item()) == 0:
            print('Ham')
        elif round(y_pred.item()) == 1:
            print('Spam')
