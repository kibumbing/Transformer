from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import TransformerModel

rawdata = pd.read_csv("cossin.csv", encoding='CP949')

# plt.figure(figsize=(20,5))
# plt.plot(range(len(rawdata)), rawdata['result'])
# plt.show()
# print(rawdata)

train = rawdata[:-100]
data_train = train["result"].to_numpy()

test = rawdata[-100:]
data_test = test["result"].to_numpy()

from torch.utils.data import DataLoader, Dataset

class cossinDataset(Dataset):
    def __init__(self, y, input_window=19, output_window=1, stride=1):
        L = y.shape[0]
        num_samples = L - input_window - output_window

        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride * i
            end_x = start_x + input_window
            X[:, i] = y[start_x:end_x]

            start_y = stride * i + input_window
            end_y = start_y + output_window
            Y[:, i] = y[start_y:end_y]
        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1, 0, 2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1, 0, 2))
        self.x = X
        self.y = Y

        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len

train_dataset = cossinDataset(data_train, input_window=19, output_window=1, stride=1)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers= 2)

lr = 1e-4
model = TransformerModel(1, 512, 8, 19, 4, 0.1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epoch = 1
model.train()
progress = tqdm(range(epoch))
for i in progress:
    batchloss = 0.0
    for (inputs, outputs) in train_loader:
        optimizer.zero_grad()
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1])
        result = model(inputs.float(), src_mask)
        loss = criterion(result, outputs[:,:,0].float())
        loss.backward()
        optimizer.step()
        batchloss += loss
    progress.set_description("loss: {:0.6f}".format(batchloss.item() / len(train_loader)))

def evaluate():
    input = [data_test[i : i+19] for i in range(0, len(data_test) - 19)]
    input = torch.tensor(input).reshape(-1, 19, 1).float()
    model.eval()

    src_mask = model.generate_square_subsequent_mask(input.shape[1])
    predictions = model(input, src_mask)
    return predictions.detach().cpu().numpy()

result = evaluate()
real = rawdata["result"].to_numpy()

plt.figure(figsize=(20,5))
plt.plot(range(1700,1999), real[1700:], label="real")
plt.plot(range(1999-81, 1999), result, label="predict")
plt.legend()
plt.show()
plt.savefig('consin1.png')