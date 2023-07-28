import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
from tqdm import tqdm
import sys
from dataload import load_data
from model import FCNet


def shuffle_and_minibatch(dataset, batch_size, shuffle=True):
    num_data = dataset['x'].shape[0]
    if shuffle:
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        shuffle_x = dataset['x'][idx]
        shuffle_y = dataset['y'][idx]
    else:
        shuffle_x = dataset['x']
        shuffle_y = dataset['y']
    mini_batches = [[shuffle_x[k:min(k + batch_size, num_data)], shuffle_y[k:min(k + batch_size, num_data)]] for k in
                    range(0, num_data, batch_size)]
    return mini_batches


def train_model(data_name, portion, num_epoch=1000, if_writer=False):
    dataset = load_data(name=data_name)
    idx = np.arange(dataset['x'].shape[0])
    np.random.shuffle(idx)
    dataset['x'] = dataset['x'][idx]
    dataset['y'] = dataset['y'][idx]
    dataset['z'] = [dataset['z'][i] for i in idx]

    num_data = dataset['x'].shape[0]
    dataset_trian = {}
    dataset_test = {}

    dataset_trian['x'] = dataset['x'][:int(num_data*portion)]
    dataset_trian['y'] = dataset['y'][:int(num_data*portion)]
    dataset_test['x'] = dataset['x'][int(num_data*portion):]
    dataset_test['y'] = dataset['y'][int(num_data*portion):]
    dataset_test['z'] = dataset['z'][int(num_data * portion):]

    # check gpu acceleration availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    batch_size = 32

    # instantiate LeNet model
    net = FCNet() #FIXME FC_NET
    # path to save the model
    PATH = './' + data_name + '_' + net.name + '_001_small_decouple_new.pth'
    # print model summary
    if if_writer:
        print(summary(net, torch.zeros((1, 19, 5))))
    # send model to GPU
    net.to(device)
    # set up loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 500], gamma=0.1)

    best_val_f1 = 0
    if if_writer:
        writer = SummaryWriter(comment=data_name+'_' + net.name + '_001_small_decouple')

    # train model
    for epoch in range(num_epoch):
        net.train()
        running_loss = 0.
        running_recall = 1.
        running_acc = 1.

        mini_batches = shuffle_and_minibatch(dataset_trian, batch_size)

        with tqdm(mini_batches, unit="batch", file=sys.stdout) as tepoch:
            for i, data in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")

                # get the inputs; data is a list of [inputs, labels]
                bbox = torch.from_numpy(data[0]).to(device)
                label = torch.from_numpy(data[1]).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(bbox)
                loss = criterion(outputs, label)

                predictions = (outputs > (torch.zeros_like(outputs) + 0.5)).long().cpu().numpy()
                label = label.cpu().numpy()

                TP = (np.logical_and(predictions == label, label == 1)).sum()
                recall = TP / label.sum()
                acc = TP / (predictions.sum() + 1)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_recall = (recall + i * running_recall) / (i + 1)
                running_acc = (acc + i * running_acc) / (i + 1)
                log = OrderedDict()
                log['loss'] = running_loss
                log['acc'] = running_acc
                log['recall'] = running_recall
                tepoch.set_postfix(log)

            scheduler.step()

            # validation
            predictions = []
            net.eval()
            with torch.no_grad():
                total = 0
                val_loss = 0
                mini_batches = shuffle_and_minibatch(dataset_test, 100, shuffle=False)
                for data in mini_batches:
                    bbox = torch.from_numpy(data[0]).to(device)
                    label = torch.from_numpy(data[1]).to(device)

                    outputs = net(bbox)
                    val_loss += nn.BCELoss(reduction='sum')(outputs, label).item()
                    total += torch.numel(label)
                    prediction = (outputs > (torch.zeros_like(outputs) + 0.5)).long().cpu().numpy()
                    predictions.append(prediction)
                val_loss /= total
                predictions = np.concatenate(predictions, 0)

                TP = (np.logical_and(predictions == dataset_test['y'], dataset_test['y'] == 1)).sum()
                val_acc = TP / (predictions.sum() + 1)
                val_recall = TP / dataset_test['y'].sum()

                print('val_loss={:.8f}, val_acc={:.8f}, val_recall={:.8f}'.format(val_loss, val_acc, val_recall), flush=True)

        if if_writer:
            writer.add_scalar('Loss/train', running_loss, epoch)
            writer.add_scalar('Loss/test', val_loss, epoch)
            writer.add_scalar('acc/train', running_acc, epoch)
            writer.add_scalar('acc/test', val_acc, epoch)
            writer.add_scalar('recall/train', running_recall, epoch)
            writer.add_scalar('recall/test', val_recall, epoch)

    if if_writer:
        writer.close()
    torch.save(net.state_dict(), PATH)
    print('Finished Training')

    # load the best model
    net.load_state_dict(torch.load(PATH))
    net.eval()
    # test
    predictions = []
    raw_predictions = []
    with torch.no_grad():
        total = 0
        val_loss = 0
        mini_batches = shuffle_and_minibatch(dataset_test, 100, shuffle=False)
        for i, data in enumerate(mini_batches):
            bbox = torch.from_numpy(data[0]).to(device)
            label = torch.from_numpy(data[1]).to(device)

            outputs = net(bbox)
            val_loss += nn.BCELoss(reduction='sum')(outputs, label).item()
            total += torch.numel(label)
            prediction = (outputs > (torch.zeros_like(outputs) + 0.5)).long().cpu().numpy()
            predictions.append(prediction)
            raw_predictions.append(outputs.cpu().numpy())
        val_loss /= total
        predictions = np.concatenate(predictions, 0)
        raw_predictions = np.concatenate(raw_predictions, 0)

        eval = {}
        TP = (np.logical_and(predictions == dataset_test['y'], dataset_test['y'] == 1)).sum()
        val_acc = TP / (predictions.sum() + 1)
        val_recall = TP / dataset_test['y'].sum()
        eval['loss'] = val_loss
        eval['acc'] = val_acc
        eval['recall'] = val_recall

        print('val_loss={:.8f}, val_acc={:.8f}, val_recall={:.8f}'.format(val_loss, val_acc, val_recall), flush=True)
        predictions = [np.where(predictions[i, ...] == 1)[0] for i in range(predictions.shape[0])]
        raw_predictions = [raw_predictions[i, :] for i in range(raw_predictions.shape[0])]

    return val_loss, val_acc, val_recall


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    num_epoch = 550
    val_loss, val_acc, val_recall = train_model('cam5', 0.8, num_epoch=num_epoch, if_writer=True)
   



