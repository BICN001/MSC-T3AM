import numpy as np
import torch
import joblib
from net_train import train, KD_train
from PreProcess import butter_bandpass_filter, seed_torch

with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

if __name__ == "__main__":
    seed_torch(2024)
    for i in range(1, 29):
        path = './lower_data/sub{}.pkl'.format(i)
        with open(path, 'rb') as f:
            t = joblib.load(f)
        train_data = np.array(t[0])[:, :, 500:]
        train_label = t[1] - 1

        train_data = butter_bandpass_filter(train_data, 8, 30)
        train_data = (train_data - train_data.mean()) / train_data.std()
        mean_acc = []

        # acc = train(train_data, train_label, 5, 200, 16, device)
        acc = KD_train(train_data, train_label, 1.5, 5, 200, 16, device, i, True)
        mean_acc.append(acc)
        mean_acc = torch.tensor(mean_acc)
