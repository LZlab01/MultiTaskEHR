import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import warnings
from sklearn.utils import class_weight
from utils import setup_seed, print_metrics_binary, mre_f, device
from model import Model_Imp, Model_Pre
from data_extraction import data_process_mimic3


def train(train_loader,
          valid_loader,
          input_dim,
          hidden_dim,
          drop_prob1,
          drop_prob2,
          lr1,
          lr2,
          alpha,
          seed,
          epochs,
          file_name,
          device):

    model_imp = Model_Imp(input_dim, hidden_dim, drop_prob1).to(device)
    model_pre = Model_Pre(input_dim, hidden_dim, drop_prob2).to(device)

    opt_model_imp = torch.optim.Adam(model_imp.parameters(), lr=lr1)
    opt_model_pre = torch.optim.Adam(model_pre.parameters(), lr=lr2)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_model, milestones=[40, 60, 80, 90], gamma=0.5)

    setup_seed(seed)
    max_auroc = 0
    train_loss_ce = []
    train_loss_mae = []
    train_loss_mre = []
    valid_loss_mae = []
    valid_loss_mre = []

    for each_epoch in range(epochs):
        batch_loss_ce = []
        batch_loss_mae = []
        batch_loss_mre = []
        model_imp.train()
        model_pre.train()

        for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.to(device)
            batch_ts = batch_ts.float().to(device)
            batch_ts = batch_ts.unsqueeze(2).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])

            mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                               torch.zeros(batch_x.shape).to(device))

            x_final = model_imp(batch_x, mask, batch_ts)
            mae_f = torch.nn.L1Loss(reduction='mean')
            mae_loss = mae_f(mask * x_final, mask * batch_x)
            loss_imp = mae_loss

            opt_model_imp.zero_grad()
            loss_imp.backward()
            opt_model_imp.step()

            loss_mae = mae_f(mask * x_final, mask * batch_x)
            loss_mre = mre_f(mask * x_final, mask * batch_x)
            batch_loss_mae.append(loss_mae.cpu().detach().numpy())
            batch_loss_mre.append(loss_mre.cpu().detach().numpy())

            output = model_pre(batch_x, mask, batch_ts, sorted_length)

            batch_y = batch_y.long()
            y_out = batch_y.cpu().numpy()
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_out),
                                                              y=y_out)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            ce_f = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

            loss_ce = ce_f(output, batch_y)
            batch_loss_ce.append(loss_ce.cpu().detach().numpy())

            i = 0
            reg_loss = None
            param_imp = list(model_imp.parameters())
            for param in model_pre.parameters():
                param_reg = param - param_imp[i]
                if reg_loss is None:
                    reg_loss = param_reg.norm(np.inf)
                else:
                    reg_loss = reg_loss + param_reg.norm(np.inf)
                if i == len(param_imp) - 3:
                    break
                i += 1

            loss_pre = loss_ce + alpha * reg_loss
            opt_model_pre.zero_grad()
            loss_pre.backward()
            opt_model_pre.step()

        train_loss_ce.append(np.mean(np.array(batch_loss_ce)))
        train_loss_mae.append(np.mean(np.array(batch_loss_mae)))
        train_loss_mre.append(np.mean(np.array(batch_loss_mre)))
        # scheduler.step()

        with torch.no_grad():
            y_true = []
            y_pred = []
            batch_loss_mae = []
            batch_loss_mre = []
            model_imp.eval()
            model_pre.eval()

            for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.to(device)
                batch_ts = batch_ts.float().to(device)
                batch_ts = batch_ts.unsqueeze(2).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])

                mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                                   torch.zeros(batch_x.shape).to(device))

                x_final = model_imp(batch_x, mask, batch_ts)
                loss_mae = mae_f(mask * x_final, mask * batch_x)
                loss_mre = mre_f(mask * x_final, mask * batch_x)
                batch_loss_mae.append(loss_mae.cpu().detach().numpy())
                batch_loss_mre.append(loss_mre.cpu().detach().numpy())

                output = model_pre(batch_x, mask, batch_ts, sorted_length)
                output = F.softmax(output, dim=1)

                batch_y = batch_y.long()
                y_pred.append(output)
                y_true.append(batch_y)

        y_pred = torch.cat(y_pred, 0)
        y_true = torch.cat(y_true, 0)
        valid_y_pred = y_pred.cpu().detach().numpy()
        valid_y_true = y_true.cpu().detach().numpy()
        ret = print_metrics_binary(valid_y_true, valid_y_pred)

        valid_loss_mae.append(np.mean(np.array(batch_loss_mae)))
        valid_loss_mre.append(np.mean(np.array(batch_loss_mre)))

        cur_mae = valid_loss_mae[-1]
        cur_mre = valid_loss_mre[-1]
        cur_auroc = ret['auroc']
        cur_auprc = ret['auprc']

        if cur_auroc > max_auroc:
            best_epoch = each_epoch
            max_auroc = cur_auroc
            state = {
                'net_imp': model_imp.state_dict(),
                'optimizer_imp': opt_model_imp.state_dict(),
                'net_pre': model_pre.state_dict(),
                'optimizer_pre': opt_model_pre.state_dict(),
                'epoch': each_epoch
            }
            torch.save(state, file_name)

    return best_epoch


def test(test_loader,
         input_dim,
         hidden_dim,
         drop_prob1,
         drop_prob2,
         seed,
         file_name,
         device):

    setup_seed(seed)
    model_imp = Model_Imp(input_dim, hidden_dim, drop_prob1).to(device)
    model_pre = Model_Pre(input_dim, hidden_dim, drop_prob2).to(device)
    checkpoint = torch.load(file_name)
    model_imp.load_state_dict(checkpoint['net_imp'])
    model_pre.load_state_dict(checkpoint['net_pre'])
    model_imp.eval()
    model_pre.eval()

    batch_loss_mae = []
    batch_loss_mre = []
    test_loss_mae = []
    test_loss_mre = []
    y_true = []
    y_pred = []
    for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.to(device)
        batch_ts = batch_ts.float().to(device)
        batch_ts = batch_ts.unsqueeze(2).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])

        mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                           torch.zeros(batch_x.shape).to(device))

        x_final = model_imp(batch_x, mask, batch_ts)

        mae_f = torch.nn.L1Loss(reduction='mean')
        loss_mae = mae_f(mask * x_final, mask * batch_x)
        loss_mre = mre_f(mask * x_final, mask * batch_x)
        batch_loss_mae.append(loss_mae.cpu().detach().numpy())
        batch_loss_mre.append(loss_mre.cpu().detach().numpy())

        output = model_pre(batch_x, mask, batch_ts, sorted_length)
        output = F.softmax(output, dim=1)

        batch_y = batch_y.long()
        y_pred.append(output)
        y_true.append(batch_y)


    y_pred = torch.cat(y_pred, 0)
    y_true = torch.cat(y_true, 0)
    test_y_pred = y_pred.cpu().detach().numpy()
    test_y_true = y_true.cpu().detach().numpy()
    ret = print_metrics_binary(test_y_true, test_y_pred)
    cur_auroc = ret['auroc']
    cur_auprc = ret['auprc']

    test_loss_mae.append(np.mean(np.array(batch_loss_mae)))
    test_loss_mre.append(np.mean(np.array(batch_loss_mre)))
    cur_mae = test_loss_mae[-1]
    cur_mre = test_loss_mre[-1]

    results = {'auroc': cur_auroc, 'auprc': cur_auprc, 'mae': cur_mae, 'mre': cur_mre}

    return results


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--drop_prob1", type=float)
    parser.add_argument("--drop_prob2", type=float)
    parser.add_argument("--lr1", type=float)
    parser.add_argument("--lr2", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--time_length", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--file_name", type=str)
    args = parser.parse_args()

    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    drop_prob1 = args.drop_prob1
    drop_prob2 = args.drop_prob2
    lr1 = args.lr1
    lr2 = args.lr2
    alpha = args.alpha
    seed = args.seed
    epochs = args.epochs
    time_length = args.time_length
    data_path = args.data_path
    file_name = args.file_name

    train_loader, valid_loader, test_loader = data_process_mimic3(data_path, time_length)
    best_epoch = train(train_loader, valid_loader, input_dim, hidden_dim, drop_prob1, drop_prob2, lr1, lr2, alpha, seed, epochs, file_name, device)
    results = test(test_loader, input_dim, hidden_dim, drop_prob1, drop_prob2, seed, file_name, device)
    print(results)

