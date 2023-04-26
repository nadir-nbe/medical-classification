import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time,sys
from model import SentimentClassifier
from dataloader import SSTDataset
import pdb
import glob

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def logit(x):
    """Compute softmax values for each sets of scores in x."""
    return np.log(x) + np.log(np.ones(len(x))-x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Function to calculate the accuracy of our predictions vs labels

def transform_logits_sigmoid(preds, labels):
    pred = sigmoid(preds)
    pred = (pred > 0.5).astype(float)

    return labels, pred


def flat_accuracy(preds, labels):
    labels_flat, pred_flat = transform_logits_sigmoid(preds, labels)
    labels_flat = labels.flatten()
    pred_flat = pred_flat.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def array_per_class(array, class_number):
    columns = [class_number]
    return array[:, columns]


def tps(z):
    return np.sum(z == 1, axis=0)


def fps(z):
    return np.sum(z == -1, axis=0)


def fns(z):
    return np.sum(z == 2, axis=0)


def tns(z):
    return np.sum(z == 0, axis=0)


def compute_2_label_pred(preds, labels):
    labels_flat, pred_flat = transform_logits_sigmoid(preds, labels)
    return 2 * labels_flat - pred_flat


def micro_recall(tps, fns):
    tps_sum = np.sum(tps)
    fns_sum = np.sum(fns)
    return tps_sum / (tps_sum + fns_sum)


def micro_precision(tps, fps):
    tps_sum = np.sum(tps)
    fps_sum = np.sum(fps)
    return tps_sum / (tps_sum + fps_sum)


def normal_recall(tp, fn):
    return tp / (tp + fn)


def normal_precision(tp, fp):
    return tp / (tp + fp)


def negative_stats(tns, fps):
    tns_sum = np.sum(tns)
    fps_sum = np.sum(fps)
    return tns_sum / (tns_sum + fps_sum)


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    # print("probs",probs)
    # print("labels",labels)
    # print("soft_probs",soft_probs)
    # print("(soft_probs.squeeze() == labels)",(soft_probs.squeeze() == labels))
    # print("(soft_probs.squeeze() == labels).float()",(soft_probs.squeeze() == labels).float())
    # print("(soft_probs.squeeze() == labels).float().mean()",(soft_probs.squeeze() == labels).float().mean())
    # print("labels",labels)
    # print("soft_probs",soft_probs.squeeze())
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

from sklearn.metrics import f1_score
def evaluate(net, criterion, dataloader, args):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0
    epsilon = sys.float_info.epsilon
    total_train_tps = np.array([0, 0, 0, 0, 0])
    total_train_tns = np.array([0, 0, 0, 0, 0])
    total_train_fns = np.array([epsilon, epsilon, epsilon, epsilon,epsilon])
    total_train_fps = np.array([epsilon, epsilon, epsilon, epsilon,epsilon])
    labous = []
    lagout = []
    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(args['gpu']), attn_masks.cuda(args['gpu']), labels.cuda(args['gpu'])
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1
            logits_cpu = logits.detach().cpu().numpy()
            label_ids_cpu = labels.to('cpu').numpy()
            z = compute_2_label_pred(logits_cpu, label_ids_cpu)
            total_train_tps += tps(z)
            total_train_fns += fns(z)
            total_train_fps += fps(z)
            total_train_tns += tns(z)

            lab, log = transform_logits_sigmoid(logits_cpu, label_ids_cpu)
            labous.append((lab>0).astype(int))
            lagout.append((log>0).astype(int))
    labous = np.concatenate(labous, axis=0)
    lagout = np.concatenate(lagout, axis=0)
    real_token_predictions = []
    real_token_labels = []

    # For each of the input tokens in the dataset...
    for i in range(len(labous)):
        real_token_predictions.append(lagout)
        real_token_labels.append(labous)
    #print("real_token_predictions",real_token_predictions)
    #print("real_token_labels",real_token_labels)
    #f1 = f1_score(real_token_labels, real_token_predictions, average='micro')

    #print("F1 score ",f1)

    print("total_train_tps", total_train_tps)
    print("total_train_fns", total_train_fns)
    print("total_train_fps", total_train_fps)
    print("total_train_tns", total_train_tns)
    print("micro_recall", micro_recall(total_train_tps, total_train_fns))
    print("micro_precision", micro_precision(total_train_tps, total_train_fps))
    print("micro F1",
          2 * (micro_recall(total_train_tps, total_train_fns) * micro_precision(total_train_tps, total_train_fps)) / (
                  micro_recall(total_train_tps, total_train_fns) + micro_precision(total_train_tps,
                                                                                   total_train_fps)))
    print("negative_stats", negative_stats(total_train_tns, total_train_fps))
    recall_class_1 = normal_recall(total_train_tps[0], total_train_fns[0])
    recall_class_2 = normal_recall(total_train_tps[1], total_train_fns[1])
    recall_class_3 = normal_recall(total_train_tps[2], total_train_fns[2])
    recall_class_4 = normal_recall(total_train_tps[3], total_train_fns[3])
    recall_class_5 = normal_recall(total_train_tps[4], total_train_fns[4])
    # recall_class_6 = normal_recall(total_train_tps[5], total_train_fns[5])

    precision_class_1 = normal_precision(total_train_tps[0], total_train_fps[0])
    precision_class_2 = normal_precision(total_train_tps[1], total_train_fps[1])
    precision_class_3 = normal_precision(total_train_tps[2], total_train_fps[2])
    precision_class_4 = normal_precision(total_train_tps[3], total_train_fps[3])
    precision_class_5 = normal_precision(total_train_tps[4], total_train_fps[4])
    # precision_class_6 = normal_precision(total_train_tps[5], total_train_fps[5])

    print("normal recalls", recall_class_1, recall_class_2, recall_class_3, recall_class_4,recall_class_5)
    print("normal precisions", precision_class_1, precision_class_2, precision_class_3, precision_class_4,precision_class_5)
    print("Macro recall avg",
          (recall_class_1 + recall_class_2 + recall_class_3 + recall_class_4+recall_class_5) / 5)
    print("Macro precision avg", (
            precision_class_1 + precision_class_2 + precision_class_3 + precision_class_4+precision_class_5) / 5)

    return mean_acc / count, mean_loss / count



def train(net, criterion, opti, train_loader, val_loader, args):

    best_acc = 0
    for ep in range(args['max_eps']):
        
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()  
            #Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(args['gpu']), attn_masks.cuda(args['gpu']), labels.cuda(args['gpu'])

            #Obtaining the logits from the model
            logits = net(seq, attn_masks)


            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            #Backpropagating the gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            #Optimization step
            opti.step()
            scheduler.step()
            if it % args['print_every'] == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it, ep, loss.item(), acc))
                #print("logits2", torch.sigmoid (logits.squeeze(-1)))
                #print("labels.float()", labels.float())



        val_acc, val_loss = evaluate(net, criterion, val_loader, args)
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep, val_acc, val_loss))
        #if val_acc > best_acc:
        print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
        best_acc = val_acc
        torch.save(net.state_dict(), './Models/clinical_{}_freeze_{}.dat'.format(ep, args['freeze_bert']))
        #net.bert_layer.save_pretrained('../Models/Pretrained_{}/'.format(ep))



if __name__ == "__main__":

    args = {}
    args['gpu'] = 0
    args['freeze_bert'] = False
    args['maxlen'] = 100
    args['batch_size'] = 8
    args['lr'] = 2e-5
    args['print_every'] = 100
    args['max_eps'] = 15

    #Instantiating the classifier model
    print("Building model! (This might take time if you are running this for first time)")
    st = time.time()
    net = SentimentClassifier(args['freeze_bert'])
    #net.load_state_dict(torch.load("../Models/best-ever.dat"))
    net.cuda(args['gpu']) #Enable gpu support for the model
    print("Done in {} seconds".format(time.time() - st))

    total = 6630 + 337 + 6242 + 1110
    weight_init = [6630 / total, 337 / (total), 6242 / total, 1110 / (total)]
    print(weight_init)
    weight = torch.tensor(
        [softmax(weight_init)]).cuda()
    print(weight)
    weight = torch.tensor(
        [np.exp(weight_init)]).cuda()
    print(weight)
    weight = torch.tensor(
        [np.exp(np.exp(np.exp(weight_init)))]).cuda()
    print(weight)
    # exit(0)
    pos_weight = torch.tensor(
        [(total - 13396) / 13396, (total - 1378) / 1378, (total - 7376) / 7376, (total - 412) / 412,
         (total - 6863) / 6863, (total - 1247) / 1247]).cuda()
    pos_weight = torch.tensor(
        [(total) / 6866, (total) / 408, (total) / 6455, (total) / 1126]).cuda()

    pos_weight = torch.tensor(
        [1.0, 3.0,  1.0, 3.0]).cuda()
    pos_weight = torch.tensor(
        [(6866) / 6866, (6866) / 408, (6866) / 6455, (6866) / 1126]).cuda()
    print("Creating criterion and optimizer objects")
    st = time.time()
    criterion = nn.BCEWithLogitsLoss()
    #opti = optim.Adam(net.parameters(), lr = args['lr'],weight_decay=1e-4)
    opti = optim.Adam(net.parameters(), lr = args['lr'])


    print("Done in {} seconds".format(time.time() - st))

    #Creating dataloaders
    print("Creating train and val dataloaders")
    st = time.time()
    train_set = SSTDataset(filename = './data_results/results_train.txt', maxlen = args['maxlen'])
    val_set = SSTDataset(filename = './data_results/results_val.txt', maxlen = args['maxlen'])
    print(train_set.__getitem__(2))

    train_loader = DataLoader(train_set, batch_size = args['batch_size'], num_workers = 5)
    val_loader = DataLoader(val_set, batch_size = 100, num_workers = 5)
    print("Done in {} seconds".format(time.time() - st))

    from transformers import get_linear_schedule_with_warmup

    total_steps = len(train_loader) * 10
    print("total_steps",total_steps)
    scheduler = get_linear_schedule_with_warmup(opti, num_warmup_steps=total_steps, num_training_steps=total_steps)



    print("Let the training begin")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    st = time.time()

    train(net, criterion, opti, train_loader, val_loader, args)
    print("Done in {} seconds".format(time.time() - st))


