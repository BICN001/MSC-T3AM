import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from Model.MSC_T3AM import MSC_T3AM
from braindecode.models import EEGNetv4

class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        self.feature = torch.tensor(self.feature, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.feature[item], self.label[item]

def train(datas, labels, fold, epoch, batch_size, device, is_change_data=True):

    if is_change_data:
        datas = datas.copy()
        datas = torch.tensor(datas).type(torch.FloatTensor).to(device)
        labels = torch.tensor(labels).type(torch.LongTensor).to(device)

    cv = KFold(fold, shuffle=False)
    cv_split = cv.split(datas)
    mean_acc = []

    for id, (train_id, test_id) in enumerate(cv_split):
        net = MSC_T3AM(10, 3, 6, 62, 1500).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        train_set = Dataset(datas[train_id], labels[train_id])
        test_set = Dataset(datas[test_id], labels[test_id])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_run_max = 0
        train_step = 0

        for i in range(epoch):
            net.train()
            train_running_acc = 0
            total = 0
            loss_steps = []
            for (x, y) in train_loader:
                x = x.to(device)
                y = y.to(device)
                out = net(x)
                loss = criterion(out, y.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_steps.append(loss.item())
                total += y.shape[0]
                pred = out.argmax(dim=1, keepdim=True)
                train_running_acc += pred.eq(y.view_as(pred)).sum().item()

            train_running_loss = float(np.mean(loss_steps))
            train_running_acc = 100 * train_running_acc / total

            train_step += 1
            if (train_step + 1) % 1 == 0:
                net.eval()
                test_running_acc = 0
                total = 0
                loss_steps = []
                with torch.no_grad():
                    for (x2, y2) in test_loader:
                        x2 = x2.to(device)
                        y2 = y2.to(device)
                        out2 = net(x2)
                        loss = criterion(out2, y2.long())

                        loss_steps.append(loss.item())
                        total += y2.shape[0]
                        pred = out2.argmax(dim=1, keepdim=True)
                        test_running_acc += pred.eq(y2.view_as(pred)).sum().item()

                    test_running_acc = 100 * test_running_acc / total
                    test_running_loss = float(np.mean(loss_steps))

                    if test_running_acc > test_run_max:
                        test_run_max = test_running_acc

            lrStep.step()
        mean_acc.append(test_run_max)

    mean_acc = torch.tensor(mean_acc)
    mean_acc = mean_acc.mean()
    return mean_acc


def KD_train(datas, labels, temp_loss, fold, epoch, batch_size, device, person, on_KD=True, is_change_data=True):
    if is_change_data:
        datas = datas.copy()
        datas = torch.tensor(datas).type(torch.FloatTensor).to(device)
        labels = torch.tensor(labels).type(torch.LongTensor).to(device)

    cv = KFold(fold, shuffle=False)
    cv_split = cv.split(datas)
    mean_acc = []

    for id, (train_id, test_id) in enumerate(cv_split):
        net_t = EEGNetv4(62, 6, 1500).to(device)
        if (on_KD == False):
            net_t.load_state_dict(torch.load('./V4/sub{}_{}.pt'.format(person, id)))
        net_s = MSC_T3AM(10, 3, 6, 62, 1500).to(device)
        optimizer_t = torch.optim.Adam(net_t.parameters(), lr=0.01, weight_decay=0.01)
        optimizer_s = torch.optim.Adam(net_s.parameters(), lr=0.01, weight_decay=0.01)
        criterion_CE = nn.CrossEntropyLoss()
        criterion_KLD = nn.KLDivLoss()
        lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=30)

        train_set = Dataset(datas[train_id], labels[train_id])
        test_set = Dataset(datas[test_id], labels[test_id])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

        test_run_max = 0
        train_step = 0

        for i in range(epoch):
            if on_KD:
                net_t.train()
            else:
                net_t.eval()
            net_s.train()
            train_running_acc_t = 0
            train_running_acc_s = 0
            total = 0
            loss_t_steps = []
            loss_s_steps = []
            for (x, y) in train_loader:
                x = x.to(device)
                y = y.to(device)
                out_t = net_t(x)
                out_s = net_s(x)
                if on_KD:
                    loss_t = criterion_CE(out_t, y.long())
                    optimizer_t.zero_grad()
                    loss_t.backward()
                    optimizer_t.step()
                    loss_t_steps.append(loss_t.item())

                loss_s = criterion_CE(out_s, y.long()) + criterion_KLD(
                    nn.functional.log_softmax(out_s / temp_loss, dim=1),
                    nn.functional.softmax(out_t.detach() / temp_loss, dim=1))
                optimizer_s.zero_grad()
                loss_s.backward()
                optimizer_s.step()

                loss_s_steps.append(loss_s.item())
                total += y.shape[0]
                pred_t = out_t.argmax(dim=1, keepdim=True)
                pred_s = out_s.argmax(dim=1, keepdim=True)
                train_running_acc_t += pred_t.eq(y.view_as(pred_t)).sum().item()
                train_running_acc_s += pred_s.eq(y.view_as(pred_s)).sum().item()

            if on_KD:
                train_running_loss_t = float(np.mean(loss_t_steps))
            train_running_loss_s = float(np.mean(loss_s_steps))
            train_running_acc_t = 100 * train_running_acc_t / total
            train_running_acc_s = 100 * train_running_acc_s / total

            train_step += 1
            if (train_step + 1) % 1 == 0:
                net_t.eval()
                net_s.eval()
                test_running_acc_t = 0
                test_running_acc_s = 0
                total = 0
                loss_t_steps = []
                loss_s_steps = []
                with torch.no_grad():
                    for (x2, y2) in test_loader:
                        x2 = x2.to(device)
                        y2 = y2.to(device)
                        out2_t = net_t(x2)
                        out2_s = net_s(x2)
                        loss_t = criterion_CE(out2_t, y2.long())
                        loss_s = criterion_CE(out2_s, y2.long())
                        loss_t_steps.append(loss_t.item())
                        loss_s_steps.append(loss_s.item())
                        total += y2.shape[0]
                        pred_t = out2_t.argmax(dim=1, keepdim=True)
                        pred_s = out2_s.argmax(dim=1, keepdim=True)

                        test_running_acc_t += pred_t.eq(y2.view_as(pred_t)).sum().item()
                        test_running_acc_s += pred_s.eq(y2.view_as(pred_s)).sum().item()

                    test_running_acc_t = 100 * test_running_acc_t / total
                    test_running_acc_s = 100 * test_running_acc_s / total
                    test_running_loss_t = float(np.mean(loss_t_steps))
                    test_running_loss_s = float(np.mean(loss_s_steps))

                    if test_running_acc_s > test_run_max:
                        test_run_max = test_running_acc_s

            lrStep.step()
        mean_acc.append(test_run_max)

    mean_acc = torch.tensor(mean_acc)
    mean_acc = mean_acc.mean()
    return mean_acc
