#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    deepwalk.py
  Author:       locke
  Date created: 2018/5/8 上午10:03

  Modified by Andy Zhao 2019/11/2
"""

import time
import random
import numpy as np
from gensim.models import Word2Vec
import scipy.io as sio 

random.seed(2018)
np.random.seed(2018)


def gen_deep_walk_feature(A, number_walks=2, alpha=0, walk_length=5, window=2, workers=2, size=30):#,row, col,
    row,col = A.nonzero()
    print(row,col)
    print(row.shape)
    print(col.shape)
    edges = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1).astype(dtype=np.dtype(str))
    print("build adj_mat")
    t1 = time.time()
    G = {}
    print(edges)
    for [i, j] in edges:
        if i not in G:
            G[i] = []
        if j not in G:
            G[j] = []
        G[i].append(j)
        G[j].append(i)
    print()
    for node in G:
        # print(node)
        G[node] = list(sorted(set(G[node])))
        if node in G[node]:
            G[node].remove(node)
    # print(G,"________")

    nodes = list(sorted(G.keys()))
    print("len(G.keys()):", len(G.keys()), "\tnode_num:", A.shape[0])
    corpus = []  # 存放上下文的 list,每一个节点一个上下文(随机游走序列)
    for cnt in range(number_walks):
        random.shuffle(nodes)
        for idx, node in enumerate(nodes):
            path = [node]  # 对每个节点找到他的游走序列.
            while len(path) < walk_length:
                cur = path[-1]  # 每次从序列的尾记录当前游走位置.
                if len(G[cur]) > 0:
                    if random.random() >= alpha:
                        path.append(random.choice(G[cur]))  # 如果有邻居,邻接矩阵里随便选一个
                    else:
                        path.append(path[0])  # 如果没有,就是自己
                else:
                    break
            corpus.append(path)
    t2 = time.time()
    print("cost: {}s".format(t2 - t1))
    print("train...")
  
    # model = Word2Vec(corpus,
    #                  vector_size=size,  # emb_size
    #                  window=window,
    #                  min_count=0,
    #                  sg=1,  # skip gram
    #                  hs=1,  # hierarchical softmax
    #                  workers=workers)
    model = Word2Vec(corpus,
                     size=30,  # emb_size
                     window=window,
                     min_count=0,
                     sg=1,  # skip gram
                     hs=1,  # hierarchical softmax
                     workers=workers)
    print(size)
    print("done.., cost: {}s".format(time.time() - t2))
    output = []
    for i in range(A.shape[0]):
        if str(i) in model.wv:  # word2vec 的输出以字典的形式存在.wv 里
            output.append(model.wv[str(i)])
        else:
            print("{} not trained".format(i))
            output.append(np.zeros(size))
    print("************************",len(output))
    return output
def readText(fileName):
    content = list()
    with open(fileName,"r",encoding="utf-8") as fp:
        for line in fp:
            content.append(line.strip("\n"))
    return content
# def readText(fileName):
#     content = list()
#     with open(fileName,"r",encoding="utf-8") as fp:
#         for line in fp:
#             content.append(line.split()[0])
#     return content
def save2mat(fileName,content):
    sio.savemat(fileName,content)
if __name__=="__main__":
    # tw_fs
    # data=open('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/fs/fs_tongzhi.txt','r',encoding='utf-8')
    # DBUserEntitiesFileName=open("/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/fs/fs_USER_deepwalk.txt",'r',encoding='utf-8')  
    # data=open('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/tw/tw_tongzhi.txt','r',encoding='utf-8')
    # DBUserEntitiesFileName=open("/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/tw/tw_USER_deepwalk.txt","r",encoding='utf-8')
    
    # dblp
    x,y,xy='16','17','dblp16_17'
    # x,y,xy='16','18','dblp16_18'
    # x,y,xy='16','19','dblp16_19'


    # x,y,xy='16','20','dblp16_20'
    # x,y,xy='17','18','dblp17_18'
    # x,y,xy='17','19','dblp17_19'
    # x,y,xy='17','20','dblp17_20'
    # x,y,xy='18','19','dblp18_19'
    # x,y,xy='18','20','dblp18_20'
    # x,y,xy='19','20','dblp19_20'
    # x
    # data=open('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/codes/dblp_all/'+xy+'/20'+x+'/tongzhi_id.txt','r',encoding='utf-8')
    # DBUserEntitiesFileName=open('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/codes/dblp_all/'+xy+'/20'+x+'/dblp20'+x+'_AUTHOR_deepwalk.txt',"r",encoding='utf-8')
    # y
    data=open('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/codes/dblp_all/'+xy+'/20'+y+'/tongzhi_id.txt','r',encoding='utf-8')
    DBUserEntitiesFileName=open('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/codes/dblp_all/'+xy+'/20'+y+'/dblp20'+y+'_AUTHOR_deepwalk.txt',"r",encoding='utf-8')
    
    # # tw_fs
    # all=open("/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/fs/fs_USER.txt","r",encoding='utf-8')
    # all=open("/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/tw/tw_USER.txt","r",encoding='utf-8')

    # dblp
    # x
    # all=open('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/codes/dblp_all/'+xy+'/20'+x+'/dblp20'+x+'_AUTHOR.txt',"r",encoding='utf-8')
    # # y
    all=open('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/codes/dblp_all/'+xy+'/20'+y+'/dblp20'+y+'_AUTHOR.txt',"r",encoding="utf-8")
    
    DBUserEntitiesFileName_deepwalk=[]
    for i in DBUserEntitiesFileName:
        DBUserEntitiesFileName_deepwalk.append(i.split("\n")[0])

    DBUserEntitiesFileName_=[]
    # lst_len = 4
    # lst_dict = {"fs@0" : 0, "fs@1" : 1, "fs@2" : 2, "fs@3" : 3}
    lst_dict={}
    id=0
    for i in all:
        cur=i.split("\n")[0]
        lst_dict[cur]=int(cur.split('@')[1])
        id+=1
        DBUserEntitiesFileName_.append(cur)
    lst_len=id
    lst = np.zeros((lst_len, lst_len))
    
    for l in data:
        r, c = l.split()
        # print(r,c)
        # print(lst_dict[r],lst_dict[c])
        lst[lst_dict[r]][lst_dict[c]] = 1
        lst[lst_dict[c]][lst_dict[r]] = 1
    for i in DBUserEntitiesFileName_:
        id=int(i.split('@')[1])
        lst[id][id]=1
    A=np.array(lst)
    # print(A)
    print(A.shape[0])

    platform1_vec=gen_deep_walk_feature(A)

    # for l in all:
    #     l_=l.split("\n")[0]
    #     if l_ not in DBUserEntitiesFileName_:
    #          DBUserEntitiesFileName_.append(l_)
    #          x=np.zeros(30)
    #          platform1_vec.append(x)
    # platform1Label=DBUserEntitiesFileName_
    for l in all:
        l_=l.split("\n")[0]
        if l_ not in DBUserEntitiesFileName_:
             x=np.zeros(30)
             platform1_vec.insert(int(l_),x)
    platform1Label=DBUserEntitiesFileName_
    # print(platform1Label)

    print(platform1_vec)
    # print(len(platform1_vec))
    # print(len(platform1Label))

    platform1_vec = np.array(platform1_vec)
    print(platform1_vec.shape)
    platform1dict = {"feature":platform1_vec,"label":platform1Label}
    # print(len(platform1_vec))

    # tw_fs
    # save2mat("/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/fs/30RelationsEmbedding128.mat",platform1dict)
    # save2mat("/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/tw/30RelationsEmbedding128.mat",platform1dict)
    
    # dblp
    # x
    # save2mat('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/codes/dblp_all/'+xy+'/20'+x+'/1_30RelationsEmbedding128.mat',platform1dict)
    # # y
    save2mat('/home/qzhou20194227007/UIL/Datasets/FS_TW/FS_TW/codes/dblp_all/'+xy+'/20'+y+'/1_30RelationsEmbedding128.mat',platform1dict)


   