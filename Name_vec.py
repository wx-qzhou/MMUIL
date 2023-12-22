import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from scipy import io


def read_json(file):
    with open(file, 'r') as fp:
        load_dict = json.load(fp)
        return load_dict

def read_data(file):
    names = []
    with open(file, 'rb') as fp:
        lines = fp.readlines()
        for _line in lines:
            line = _line.decode('utf8')
            temp = line.replace('\n', '').split('\t')[1]
            name = list(filter(str.isalnum, temp.lower()))
            name_str = ''.join(name)
            names.append(name_str)

    return names


def train_model(epochs, vector_size, window, bag_name):
    documents = [TaggedDocument(list(doc), [i]) for i, doc in enumerate(bag_name)]
    model = Doc2Vec(documents, epochs=epochs, vector_size=vector_size, window=window, workers=1)
    return model


if __name__ == "__main__":
    name17 = read_data(r'C:\Users\15262\Desktop\temp\social\fs_names.txt')
    names = set()
    bag_17 = dict()
    bag_name17 = set()
    word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for name in name17:
        temp = []
        for i in name:
            if i in word:
                temp.append(i)
        str_name = ''.join(temp)
        bag_17[name] = str_name
        bag_name17.add(str_name)
        if str_name == '':
            print('11111')
        for _i in str_name:
            names.add(_i)

    print(len(names))
    print(names)
    print(len(bag_name17))
    list_name = list(bag_name17)

    model = train_model(640, 30, 2, list_name)

    vec_dict = dict()
    for index, key in enumerate(list_name):
        vec_dict[key] = model.docvecs[index]

    final_vec = []
    for name in name17:
        final_vec.append(vec_dict[bag_17[name]])

    print(np.array(final_vec).shape)
    data = {
        'profile': np.array(final_vec)
    }

    io.savemat(r'C:\Users\15262\Desktop\temp\social\fs_name.mat', data)

