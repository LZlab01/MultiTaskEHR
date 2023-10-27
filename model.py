import torch
import numpy as np
import torch.nn as nn
from utils import hdl_time, data_process, device


class Conv(nn.Module):
    def __init__(self ,in_dim=17 ,dropout=0):
        super(Conv ,self).__init__()
        self.conv_layer = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size=3, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self ,x):
        x = self.dropout(x)
        x_0 = torch.zeros(x.shape[0] ,1 ,x.shape[2]).to(device)
        x_prime = torch.cat((x_0 ,x ,x_0) ,1)

        k =  self.relu(self.conv_layer(x_prime.permute(0 ,2 ,1))).permute(0 ,2 ,1)

        return k


class GRUD_Time(nn.Module):
    def __init__(self, input_size, hidden_size, drop_prob):
        super(GRUD_Time, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        self.w_dg_h = nn.Linear(input_size, hidden_size)
        self.w_xr = nn.Linear(input_size, hidden_size)
        self.w_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_xz = nn.Linear(input_size, hidden_size)
        self.w_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_xh = nn.Linear(input_size, hidden_size)
        self.w_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(p=drop_prob)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, time_info, mask):
        batch_size = x.size(0)
        time_step = x.size(1)
        feature_dim = x.size(2)
        assert (feature_dim == self.input_size)

        time_info = time_info * mask
        time_last_list = []
        for i in range(x.shape[0]):
            time_last, _ = data_process(time_info[i])
            time_last_list.append(time_last)

        time_last = torch.stack(time_last_list)
        time_info = hdl_time(time_last, self.input_size)

        gamma_h = torch.exp(
            -torch.max(torch.zeros(batch_size, time_step, self.hidden_size).to(device), self.w_dg_h(time_info)))

        x = self.dropout(x)
        scan_input = x.permute(1, 0, 2)
        initial_hidden = torch.zeros(batch_size, self.hidden_size).to(device)

        hidden_list = []
        for i in range(time_step):
            initial_hidden = self.GRUD_Unit(scan_input[i, :, :], initial_hidden, gamma_h[:, i, :])
            hidden_list.append(initial_hidden)

        hidden_states = torch.stack(hidden_list).permute(1, 0, 2)

        return hidden_states

    def GRUD_Unit(self, x, prev_hidden, gamma_h):
        h = torch.mul(gamma_h, prev_hidden)

        r = torch.sigmoid(self.w_xr(x) + self.w_hr(h))
        z = torch.sigmoid(self.w_xz(x) + self.w_hz(h))
        h_tilde = torch.tanh(self.w_xh(x) + self.w_hh(torch.mul(r, h)))

        h = torch.mul((1 - z), h) + torch.mul(z, h_tilde)

        return h


class Model_Pre(nn.Module):
    def __init__(self, input_size, hidden_size, drop_prob=0):
        super(Model_Pre, self).__init__()
        self.conv = Conv(input_size, dropout=drop_prob)
        self.grud = GRUD_Time(input_size, hidden_size, drop_prob)

        self.pre = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, mask, t, sorted_lengths):
        x_zero = nn.Parameter(torch.zeros(x.shape)).to(device)
        nn.init.zeros_(x_zero)
        x_prime = torch.where(x == -1, x_zero, x)

        x_conv = self.conv(x_prime)
        out = self.grud(x_conv, t, mask)
        out = out + x_conv

        out_list = []
        for i in range(out.shape[0]):
            idx = sorted_lengths[i] - 1
            out_list.append(out[i, idx, :])
        out_ = torch.stack(out_list)

        output = self.pre(out_)

        return output


class Model_Imp(nn.Module):
    def __init__(self, input_size, hidden_size, drop_prob=0):
        super(Model_Imp, self).__init__()
        self.conv = Conv(input_size, dropout=drop_prob)
        self.grud = GRUD_Time(input_size, hidden_size, drop_prob)

        self.pre = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, mask, t):
        x_zero = nn.Parameter(torch.zeros(x.shape)).to(device)
        nn.init.zeros_(x_zero)
        x_prime = torch.where(x == -1, x_zero, x)

        x_conv = self.conv(x_prime)
        out = self.grud(x_conv, t, mask)
        out = out + x_conv
        output = self.pre(out)

        return output

