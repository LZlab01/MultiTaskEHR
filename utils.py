import math
import torch
import random
import numpy as np
import copy
import pandas as pd
from typing import Optional
from sklearn import metrics
from torch import nn, Tensor
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_metrics_binary(y_true, predictions, verbose=0):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    f1_score=2*prec1*rec1/(prec1+rec1)
    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        print("f1_score = {}".format(f1_score))

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse,
            "f1_score":f1_score}


def data_process(input_data):
    input_data = input_data.cpu().detach().numpy()
    input_data = np.where(input_data == 0, np.nan, input_data)
    input_ffill = pd.DataFrame(input_data).fillna(method='ffill').values
    input_ffill = np.where(np.isnan(input_ffill), 0, input_ffill)
    input_bfill = pd.DataFrame(input_data).fillna(method='bfill').values
    input_bfill = np.where(np.isnan(input_bfill), 0, input_bfill)

    x_last = torch.tensor(input_ffill).to(device)
    x_next = torch.tensor(input_bfill).to(device)

    return x_last, x_next


def hdl_time(batch_ts, input_size):
    time_out = torch.cat((torch.zeros(batch_ts.shape[0], 1, input_size).to(device), batch_ts), 1)
    time_out_diff = np.diff(time_out.cpu().detach().numpy(), axis=1)
    time_out_diff[np.where(time_out_diff < 0)] = 0

    return torch.tensor(time_out_diff).to(device)


def mre_f(y_pre, y):
    return torch.sum(torch.abs(y_pre - y)) / torch.sum(torch.abs(y))

