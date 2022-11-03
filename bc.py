import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import load_iris
import pdb
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def full_gd(model, criterion, optimizer, trainloader, n_epochs=200):

    train_losses = np.zeros(n_epochs)

    for it in range(n_epochs):

        for x, y in trainloader:

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.long())
            loss.backward()
            optimizer.step()

            # outputs_test = model(X_test)
            # loss_test = criterion(outputs_test, y_test)
            train_losses[it] = loss.item()
            # test_losses[it] = loss_test.item()

        if (it + 1) % 10 == 0:
            # print(
            #     f'In this epoch {it + 1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')
            print(
                f'In this epoch {it + 1}/{n_epochs}, Training loss: {loss.item():.4f}')

    return train_losses

iris = load_iris()
X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
x, x_val, y, y_val = train_test_split(X, Y, test_size=0.33, random_state=42)

x = x.reshape(-1, x.shape[1]).astype('float32')
x_val = x_val.reshape(-1, x_val.shape[1]).astype('float32')

x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))
x_val = torch.from_numpy(x_val.astype(np.float32))
y_val = torch.from_numpy(y_val.astype(np.float32))

data_set = Data(x, y)
trainloader=DataLoader(dataset=data_set,batch_size=64)

_, input_dimension = x.shape
output_dimension = 3

model = Model(input_dimension, output_dimension)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

train_losses = full_gd(model, criterion, optimizer, trainloader)

plt.plot(train_losses, label='train loss')
# plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

"""evaluate model"""

with torch.no_grad():
    z = model(x)
    yhat = torch.max(z.data, 1)
    train_acc = np.sum(y.numpy().astype('int8') == yhat.indices.numpy().astype('int8'))/y.shape[0]

    z_val = model(x_val)
    yhat_val = torch.max(z_val.data, 1)
    pdb.set_trace()
    test_acc = np.sum(y_val.numpy().astype('int8') == yhat_val.indices.numpy().astype('int8'))/y_val.shape[0]


    print(train_acc)
    print(test_acc)