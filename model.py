import torch
import numpy as np
from data_loader import get_data


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(3000, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.LazyLinear(128)
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
    def __init__(self, lr: float, data_path: str, data_batch_size: int, epoch: int):
        self.model = MyModel()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=0.0001)
        self.train_load, self.test_load = get_data(data_path, data_batch_size)
        self.batch_size = data_batch_size
        self.epoch = epoch

        self.run()
        self.test()

    def run(self):
        for epoch in range(self.epoch):
            accuracy = 0
            ll = []
            for batch in self.train_load:
                self.criterion.zero_grad()
                x_batch, y_batch = batch

                y_pred = self.model.forward(x_batch.to(torch.float32))

                y_pred_rounded = torch.round(y_pred)
                accuracy += (y_pred_rounded == y_batch.reshape(len(y_pred), 1)).sum().item()
                accuracy = accuracy / self.batch_size

                loss = self.criterion(y_pred, y_batch.float().reshape(len(y_pred), 1))
                ll += [loss.item()]
                loss.backward()

                self.optimizer.step()
            accuracy = accuracy / self.batch_size
            if epoch % 1 == 0:
                print(f"Epoch: {epoch}, Loss: {np.mean(ll)}, Accuracy: {accuracy}")

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
