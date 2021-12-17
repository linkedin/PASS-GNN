import numpy as np
import torch
import torch.optim as optim

from math import sqrt
from time import perf_counter

from args import get_args
from model import sampleGCN
from utils import load_npz

import warnings
warnings.filterwarnings('ignore')

# Train GCN models
def train_model(model, labels, idx, args):
    # Register trainable parameters to the optimizer
    ml = list()
    for index, module in enumerate(model.W):
        if (index == 0):
            ml.append({'params': module.parameters(), 'weight_decay': model.weight_decay})
        else:
            ml.append({'params': module.parameters()})
    ml.append({'params': model.sample_W})
    ml.append({'params': model.sample_W2})
    ml.append({'params': model.sample_a})

    optimizer = optim.Adam(ml, lr=args.lr)

    start_time = perf_counter()
    patient = 0
    loss = np.inf
    train_idx = idx["train"]
    val_idx = idx["val"]
    for _ in range(args.epochs):
        total_train_loss = 0
        total_sample_loss = 0
        # Batch training
        for b in range(train_idx.shape[0] // args.batch_size):
            batch_idx = train_idx[b * args.batch_size : (b+1) * args.batch_size]
            model.train()
            optimizer.zero_grad()
            output = model(batch_idx)
            train_loss = model.calc_loss(output, labels[batch_idx])
            total_train_loss = total_train_loss + train_loss.item()
            train_loss.backward()

            # Loss for sampling probability function
            # Gradient of intermediate tensor
            chain_grad = model.X1.grad
            # Compute intermediate loss for sampling probability parameters
            sample_loss = model.sample_loss(chain_grad.detach())
            total_sample_loss = total_sample_loss + sample_loss.item()
            sample_loss.backward()

            optimizer.step()
            torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            output = model(val_idx)
            new_loss = model.calc_loss(output, labels[val_idx])
            if new_loss >= loss:
                patient = patient + 1
            else:
                patient = 0
                loss = new_loss
        if patient == args.early_stopping:
            break

    train_time = perf_counter() - start_time
    return train_time


# Test GCN models
def test_model(model, labels, idx, args):
    start_time = perf_counter()
    test_idx = idx["test"]
    model.eval()
    total_acc = 0
    for b in range(test_idx.shape[0] // args.batch_size):
        batch_idx = test_idx[b * args.batch_size : (b+1) * args.batch_size]
        output = model(batch_idx)
        np_pred = output.cpu()
        np_target = labels[batch_idx].cpu()
        acc_mic, acc_mac = model.calc_f1(np_pred, np_target)
        total_acc = total_acc + acc_mic
    total_acc = total_acc / (test_idx.shape[0] // args.batch_size)
    test_time = perf_counter() - start_time
    return total_acc, test_time


# Train, test, and compute average performance of GCN models
def run_gnn(device, args, graph, features, feat_size, labels, label_max, idx):
    trial = 3
    total_acc = 0
    total_acc2 = 0
    total_train_time = 0
    total_test_time = 0
    for _ in range(trial):
        model = sampleGCN(feat_size, label_max, args.hidden_dim, args.step_num,
                        graph, features, args.sample_scope, args.sample_num,
                        args.nonlinear, args.dropout, args.weight_decay)
        # Move to GPUs (or CPUs)
        model = model.to(device)
        model.device = device
        train_time = train_model(model, labels, idx, args)
        total_train_time = total_train_time + train_time

        mic_acc, test_time = test_model(model, labels, idx, args)
        total_acc = total_acc + mic_acc
        total_acc2 = total_acc2 + mic_acc**2
        total_test_time = total_test_time + test_time
        del model

    total_acc = total_acc/trial
    total_acc2 = sqrt(total_acc2/trial - total_acc**2)
    total_train_time = total_train_time/trial
    total_test_time = total_test_time/trial

    print("Train time: {:.4f},\tTest time: {:.4f},\tAccuracy: {:.4f},\tstd: {:.8f}".format(total_train_time, total_test_time, total_acc, total_acc2))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_args()

    graph, features, labels, label_max, idx = load_npz(device, args.data_dir, args.dataset)
    feat_size = features.size(1)
    run_gnn(device, args, graph, features, feat_size, labels, label_max, idx)


if __name__ == "__main__":
    main()

