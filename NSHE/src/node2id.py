import json


def read_json(file):
    with open(file, 'r') as fp:
        load_dict = json.load(fp)
        return load_dict


def get_name(name):
    temp = name.split('_')
    return ' '.join(temp)


if __name__ == "__main__":
    year = '2020'
    node = read_json('/data/mmwang20195227044/uilData/node_' + year + '.json')['node']
    author = read_json('/data/mmwang20195227044/uilData/author_' + year + '3.json')
    paper = read_json('/data/mmwang20195227044/uilData/paper_' + year + '3.json')
    authors = []
    papers = set()
    for key in node:
        authors.append(key)
        _papers = author[get_name(key)]
        for p in _papers:
            papers.add(p)

    papers = list(papers)

    confs = set()
    for p in papers:
        conf = paper[p]['key'].split('/')[1]
        confs.add(conf)

    confs = list(confs)

    count = 0
    nodes = []
    author_dict = {}
    paper_dict = {}
    conf_dict = {}

    for a in authors:
        author_dict[a] = count
        line = 'a@' + a + '\t' + str(count) + '\n'
        nodes.append(line)
        count = count + 1

    for p in papers:
        paper_dict[p] = count
        line = 'p@' + p + '\t' + str(count) + '\n'
        nodes.append(line)
        count = count + 1

    for c in confs:
        conf_dict[c] = count
        line = 'c@' + c + '\t' + str(count) + '\n'
        nodes.append(line)
        count = count + 1

    realtion = []
    for key in authors:
        _papers = author[get_name(key)]
        for p in _papers:
            col = paper_dict[p]
            row = author_dict[key]
            wline = str(row) + '\t' + str(col) + '\t0' +'\n'
            realtion.append(wline)

    for p in papers:
        conf = paper[p]['key'].split('/')[1]
        row = paper_dict[p]
        col = conf_dict[conf]
        wline = str(row) + '\t' + str(col) + '\t1' + '\n'
        realtion.append(wline)

    with open('/data/mmwang20195227044/new/data/dblp/'+year+'/node2id.txt', 'w') as fp:
        tmpline = str(len(nodes)) + '\n'
        fp.write(tmpline)
        fp.writelines(nodes)

    with open('/data/mmwang20195227044/new/data/dblp/'+year+'/relations.txt', 'w') as fp:
        fp.writelines(realtion)





