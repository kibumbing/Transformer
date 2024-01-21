from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import TransformerModel

rawdata = pd.read_csv("sum1.csv", encoding='CP949')

# plt.figure(figsize=(20,5))
# plt.plot(range(len(rawdata)), rawdata['result'])
# plt.show()
# print(rawdata)

train = rawdata[:-20]
data_train = train.to_numpy()

test = rawdata[-20:]
data_test = test.to_numpy()

class sumDataset(Dataset):
    def __init__(self, y, input_window=5, output_window=1, stride=1):

        num_samples = y.shape[0]

        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            X[:, i] = y[i, :-1]
            Y[:, i] = y[i, 2]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1, 0, 2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1, 0, 2))
        self.x = X
        self.y = Y

        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len

train_dataset = sumDataset(data_train, input_window=5, output_window=1)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers= 2)

lr = 1e-4
model = TransformerModel(1, 512, 8, 5, 4, 0.1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epoch = 200
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
    input = [data_test[i, :-1] for i in range(len(data_test))]
    output = [data_test[i, 2] for i in range(len(data_test))]
    input = torch.tensor(input).reshape(-1, 5, 1).float()
    model.eval()

    src_mask = model.generate_square_subsequent_mask(input.shape[1])
    predictions = model(input, src_mask)
    return predictions.detach().cpu().numpy(), output

predict, ans = evaluate()
for i in range(len(predict)):
    print(f"{i} {predict[i][0]}, {ans[i]}")