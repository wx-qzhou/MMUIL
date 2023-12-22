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


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries) #bs,n,S
        attn = self.softmax(attn) #bs,n,S
        attn = attn / torch.sum(attn, dim=0, keepdim=True) #bs,n,S
        out = self.mv(attn) #bs,n,d_model

        return out


class Encoder(nn.Module):

    def __init__(self, Input_dim=286, Hidden_dim=512, Output_dim=1000):
        super(Encoder, self).__init__()
        self.ea = ExternalAttention(Input_dim)
        self.model = nn.Sequential(
            nn.Linear(Input_dim, Hidden_dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Hidden_dim, Output_dim, bias=False),
        )

    def forward(self, x):
        x_i = self.ea(x)
        x_i = self.model(x + x_i)
        return x_i

    def embed(self, x):
        x_i = self.ea(x)
        x_i = self.model(x + x_i)
        return x_i


class Decoder(nn.Module):

    def __init__(self, Input_dim=1000, Hidden_dim=512, Output_dim=286):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(Input_dim, Hidden_dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Hidden_dim, Output_dim, bias=False),
        )

    def forward(self, x):
        x_i = self.model(x)
        return x_i

    def embed(self, x):
        x_i = self.model(x)
        return x_i


class Discriminator(nn.Module):
    def __init__(self, Input_dim=286, Hidden_dim=512, dropout=0.2, S=32, Class_num=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(Input_dim, Hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(Hidden_dim, S),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(S, Class_num),
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


def read_data(file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        res = []
        for line in lines:
            temp = line.replace('\n', '').split('\t')
            res.append((temp[0], temp[1]))
        return res


def get_loader(ori_data):
    torch_dataset = Data.TensorDataset(ori_data)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=1,
    )
    return loader


def get_matched_users(index_i, index_j, identity_data):
    s_index = []
    t_index = []
    samples = []
    if index_i == index_j:
        inter = set(identity_data[index_i]).intersection(identity_data[index_j])
        samples = random.sample(inter, int(len(inter) * 0.8))
        for user in samples:
            for sindex, sidx in enumerate(identity_data[index_i]):
                if user == sidx:
                    s_index.append(sindex)

            for tindex, tidx in enumerate(identity_data[index_j]):
                if user == tidx:
                    t_index.append(tindex)
    else:
        train_sample = read_data('//home/qzhou20194227007/UIL/Datasets/uilData/social/' + prefix + '/train.txt')
        for user in train_sample:
            for sindex, sidx in enumerate(identity_data[index_i]):
                if user[0] == sidx:
                    s_index.append(sindex)

            for tindex, tidx in enumerate(identity_data[index_j]):
                if user[1] == tidx:
                    t_index.append(tindex)

            samples.append(user[1])

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
    out = torch.sum(F.pairwise_distance(x_pj, x_pj_data, p=2)) / samples  # 欧式距离
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
            s_index = user_dict[user[0]]
            sj_vector = y_pj[s_index]
            hj = get_hit(user[1], identitys[index_j], sj_vector, pj__data, k, index_i, index_j)
            final_dict[str(k)][user[0]] = hj
            precison[index] = precison[index] + hj

    precison_list = []
    for p in precison:
        temp = p / 450
        print(temp)
        precison_list.append(temp)

    with open('/home/qzhou20194227007/UIL/MMUIL/results/'+ 'fstw_' + prefix + '.json', 'w') as fp:
        json.dump(final_dict, fp)
    
    with open('/home/qzhou20194227007/UIL/MMUIL/results/'+ 'fstw_' + prefix + '_results.json', 'w') as fp:
        json.dump(precison_list, fp)


def get_TopK(ori_data, test_set, identitys, Encoders, Decoders, k=10):
    user_dict_i = dict()
    for user in identitys[0]:
        user_dict_i[user] = len(user_dict_i)

    get_two(0, 1, ori_data, test_set, identitys, user_dict_i, Encoders, Decoders)


def get_list(loader):
    temp = list()
    for step, batch_x in enumerate(loader):
        temp.append(batch_x)
    return temp


if __name__ == '__main__':
    prefix = '3'
    name16 = read_json(r'/home/qzhou20194227007/UIL/Datasets/uilData/social/tw_node1.json')['node']
    name17 = read_json(r'/home/qzhou20194227007/UIL/Datasets/uilData/social/fs_node1.json')['node']
    data16 = sio.loadmat(r'/home/qzhou20194227007/UIL/Datasets/uilData/social/final_tw.mat')['data']
    data17 = sio.loadmat(r'/home/qzhou20194227007/UIL/Datasets/uilData/social/final_fs.mat')['data']
    source_data1 = F.normalize(torch.FloatTensor(data16), p=2, dim=0)
    source_data2 = F.normalize(torch.FloatTensor(data17), p=2, dim=0)

    test_data = read_data('/home/qzhou20194227007/UIL/Datasets/uilData/social/' + prefix + '/test.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    opt = parser.parse_args()

    cuda = True if torch.cuda.is_available() else False

    # Optimizers

    encoders = [Encoder() for _ in range(2)]
    decoders = [Decoder() for _ in range(2)]
    discriminators = [Discriminator() for _ in range(2)]
    generator_param = get_param(encoders)
    generator_param.extend(get_param(decoders))
    optimizer_E = torch.optim.RMSprop(generator_param, lr=opt.lr)
    optimizer_D = torch.optim.RMSprop(get_param(discriminators), lr=opt.lr)
    optimizer_E.zero_grad()
    optimizer_D.zero_grad()

    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    ori_datas = [data16, data17]
    all_data = [source_data1, source_data2]
    identity = [name16, name17]
    data_list = [get_list(get_loader(source_data1)), get_list(get_loader(source_data2))]
    samples_dict = dict()
    inter_list01, samples_dict['sample01'] = get_matched_users(0, 1, identity)
    match_list = {
        '0': {
            '0': get_matched_users(0, 0, identity)[0],
            '1': inter_list01,
        },
        '1': {
            '0': [inter_list01[1], inter_list01[0]],
            '1': get_matched_users(1, 1, identity)[0],
        }
    }

    for epoch in range(opt.n_epochs):
        for u in range(0, 5):  # 多少个batch
            loss_D = 0
            optimizer_D.zero_grad()
            for i in range(2):  # 遍历所有平台
                # 随机选择一个平台
                pj_num = random.randint(0, 1)
                # Configure input
                pi_data = data_list[i][random.randint(0, len(data_list[i]) - 1)][0]
                pj_data = data_list[pj_num][random.randint(0, len(data_list[pj_num]) - 1)][0]
                pi = Variable(pi_data.type(torch.FloatTensor), requires_grad=True)
                pj = Variable(pj_data.type(torch.FloatTensor), requires_grad=True)
                #  Train Discriminator
                # Generate a batch of users
                fake_pi = encoders[i](pi).detach()  # encoder ei
                fake_pj = decoders[pj_num](fake_pi).detach()  # decoder oj
                # Adversarial loss
                loss_D = loss_D - torch.mean(discriminators[pj_num](pj)) + torch.mean(discriminators[pj_num](fake_pj))

            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # Clip weights of discriminator
            for discriminator in discriminators:
                for p in discriminator.parameters():
                    p.data.clamp_(-opt.clip_value, opt.clip_value)

            print(
                "[Epoch %d/%d] [D loss: %f]"
                % (epoch, opt.n_epochs, loss_D.item())
            )

        # Train the generator every n_critic iterations
        loss_G = 0
        loss_C = 0
        loss = 0
        optimizer_E.zero_grad()
        for i in range(2):
            #  Train Generator
            # 随机选择一个平台
            _pj_num = random.randint(0, 1)
            __pi_data = data_list[i][random.randint(0, len(data_list[i]) - 1)][0]
            _pi = Variable(__pi_data.type(torch.FloatTensor), requires_grad=True)
            # Generate a batch of images
            _fake_pi = encoders[i](_pi)  # encoder ei
            _fake_pj = decoders[_pj_num](_fake_pi)  # decoder oj
            matched_users = []
            # Adversarial loss
            loss_G = loss_G - torch.mean(discriminators[_pj_num](_fake_pj))
            loss_C = loss_C + get_matched_distance(i, _pj_num, ori_datas, match_list)
        loss = loss_G + loss_C
        loss.backward(retain_graph=True)
        optimizer_E.step()

        print(
            "[Epoch %d/%d] [G loss: %f] [C loss: %f] [Loss: %f]"
            % (epoch, opt.n_epochs, loss_G.item(), loss_C.item(), loss.item()))

    get_TopK(ori_datas, test_data, identity, encoders, decoders)
