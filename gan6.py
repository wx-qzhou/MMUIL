import argparse
import torch.utils.data as Data
import random
from torch.autograd import Variable
import json
import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy import io as sio
from collections import defaultdict
import numpy as np
from torch.nn import init


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.ea = ExternalAttention(128)
        self.model = nn.Sequential(
            nn.Linear(128, 512, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024, bias=False),
        )

    def forward(self, x):
        x_i = self.ea(x)
        x_i = self.model(x+x_i)
        return x_i

    def embed(self, x):
        x_i = self.ea(x)
        x_i = self.model(x+x_i)
        return x_i


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128, bias=False),
        )

    def forward(self, x):
        x_i = self.model(x)
        return x_i

    def embed(self, x):
        x_i = self.model(x)
        return x_i


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


def get_param(model_list):
    para = []
    for model in model_list:
        para.append({'params': model.parameters()})

    return para


def read_json(file):
    with open(file, 'r') as fp:
        load_dict = json.load(fp)
        return load_dict


def get_loader(ori_data):
    torch_dataset = Data.TensorDataset(ori_data)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=1,
    )
    return loader


def get_matched_users(index_i, index_j, identity_data, test_set):
    inter = set(identity_data[index_i]).intersection(identity_data[index_j])
    if index_i == index_j:
        samples = random.sample(inter, int(len(inter) * 0.8))
    else:
        samples = list(inter - set(test_set))
    s_index = []
    t_index = []
    for user in samples:
        for sindex, sidx in enumerate(identity_data[index_i]):
            if user == sidx:
                s_index.append(sindex)

        for tindex, tidx in enumerate(identity_data[index_j]):
            if user == tidx:
                t_index.append(tindex)

    return [s_index, t_index], samples


def get_matched_distance(index_i, index_j, ori_data, match_lists):
    sample_index = match_lists[str(index_i)][str(index_j)]
    samples = len(sample_index[0])
    pi_inputs = []
    pj_inputs = []
    _pi_data = ori_data[index_i]
    _pj_data = ori_data[index_j]
    for idx in sample_index[0]:
        pi_inputs.append(_pi_data[idx])
    for idx in sample_index[1]:
        pj_inputs.append(_pj_data[idx])
    pi_inputs = np.array(pi_inputs)
    pj_inputs = np.array(pj_inputs)
    pi__data = F.normalize(torch.FloatTensor(pi_inputs), p=2, dim=0)
    pj__data = F.normalize(torch.FloatTensor(pj_inputs), p=2, dim=0)
    x_pi_data = pi__data
    x_pj_data = pj__data
    x_pi = encoders[index_i].embed(x_pi_data)
    x_pj = decoders[index_j].embed(x_pi)
    out = torch.sum(F.pairwise_distance(x_pj, x_pj_data, p=2)) / samples  # ≈∑ Ωæ‡¿Î
    return out


def get_dis(vec1, vec2):
    vec1 = torch.FloatTensor([vec1])
    vec2 = torch.FloatTensor([vec2])
    _sim = torch.sum(F.pairwise_distance(vec1, vec2, p=2))
    print(_sim)


def get_hit(t_user, _identity, s_vec, t_data, k, index_i, index_j):
    dis = dict()
    samples = samples_dict['sample'+str(index_i)+str(index_j)]
    for m_idx, key in enumerate(_identity):
        if key not in samples:
            _dis = np.linalg.norm(s_vec - t_data[m_idx])
            dis[key] = _dis
    sort_dis = sorted(dis.items(), key=lambda d: d[1])
    h = 0
    for index in range(k):
        if sort_dis[index][0] == t_user:
            hit = index
            h = (k - hit) / k
    return h


def get_two(index_i, index_j, ori_data, test_set, identitys, user_dict, Encoders, Decoders):
    years = ['2016', '2017', '2018', '2019', '2020']
    topk = [1, 5, 10, 15, 30]
    final_dict = {
        '1': dict(),
        '5': dict(),
        '10': dict(),
        '15': dict(),
        '30': dict()
    }
    pi__data = F.normalize(torch.FloatTensor(ori_data[index_i]), p=2, dim=0)
    pj__data = F.normalize(torch.FloatTensor(ori_data[index_j]), p=2, dim=0).detach().numpy()
    x_pi_data = pi__data
    x_pi = Encoders[index_i].embed(x_pi_data)
    x_pj = Decoders[index_j].embed(x_pi)
    y_pj = x_pj.detach().numpy()
    precison = [0, 0, 0, 0, 0]
    for index, k in enumerate(topk):
        for user in test_set:
            s_index = user_dict[user]
            sj_vector = y_pj[s_index]
            hj = get_hit(user, identitys[index_j], sj_vector, pj__data, k, index_i, index_j)
            final_dict[str(k)][user] = hj
            precison[index] = precison[index] + hj
    print(years[index_i], years[index_j])
    for p in precison:
        temp = p/300
        print(temp)

    with open('/home/gmjin/datas/gan/out/res_' + prefix + '_' + years[index_i] + '_' + years[index_j] + '.json', 'w') as fp:
        json.dump(final_dict, fp)


def get_TopK(ori_data, test_set, identitys, Encoders, Decoders, k=10):
    user_dict_i = dict()
    user_dict_j = dict()
    user_dict_k = dict()
    user_dict_m = dict()
    for user in identitys[0]:
        user_dict_i[user] = len(user_dict_i)
    for user in identitys[1]:
        user_dict_j[user] = len(user_dict_j)
    for user in identitys[2]:
        user_dict_k[user] = len(user_dict_k)
    for user in identitys[3]:
        user_dict_m[user] = len(user_dict_m)

    get_two(0, 1, ori_data, test_set, identitys, user_dict_i, Encoders, Decoders)
    get_two(0, 2, ori_data, test_set, identitys, user_dict_i, Encoders, Decoders)
    get_two(0, 3, ori_data, test_set, identitys, user_dict_i, Encoders, Decoders)
    get_two(0, 4, ori_data, test_set, identitys, user_dict_i, Encoders, Decoders)
    get_two(1, 2, ori_data, test_set, identitys, user_dict_j, Encoders, Decoders)
    get_two(1, 3, ori_data, test_set, identitys, user_dict_j, Encoders, Decoders)
    get_two(1, 4, ori_data, test_set, identitys, user_dict_j, Encoders, Decoders)
    get_two(2, 3, ori_data, test_set, identitys, user_dict_k, Encoders, Decoders)
    get_two(2, 4, ori_data, test_set, identitys, user_dict_k, Encoders, Decoders)
    get_two(3, 4, ori_data, test_set, identitys, user_dict_m, Encoders, Decoders)


def get_list(loader):
    temp = list()
    for step, batch_x in enumerate(loader):
        temp.append(batch_x)
    return temp
