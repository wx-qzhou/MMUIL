import json

def read_json(file):
    with open(file, 'r') as fp:
        load_dict = json.load(fp)
        return load_dict

def obtain_multi_platforms(hit_list, num_platform=3):
    num_platform = num_platform - 1
    assert num_platform >= 1 and num_platform <= len(hit_list)
    if num_platform - 1 < len(hit_list):
        hit_list = hit_list[:num_platform] 
    user_set = set(hit_list[0].keys())
    for hit in hit_list[1:]:
        user_set = user_set.intersection(set(hit.keys()))
    
    user_list = []
    for user_name in user_set:
        flag = True
        for hit in hit_list:
            if hit[user_name] == 0:
                flag = False
                break
        if flag:
            user_list.append(user_name)
    
    precision = 0
    for user_name in user_list:
        for hit in hit_list:
            precision += hit[user_name]
    print(precision / len(user_set) / num_platform)

if __name__ == "__main__":
    appix = "dblp_no_EMAT_3_"
    dir_name = "/home/qzhou20194227007/UIL/MMUIL/results/"
    file_list = ["2016", "2017", "2018", "2019", "2020"]
    hit1_list = []
    hit5_list = []
    hit10_list = []
    hit15_list = []
    hit30_list = []

    for i in range(1, len(file_list)):
        data = read_json(dir_name + appix + file_list[0] + "_" + file_list[i] + ".json")
        # print(dir_name + appix + file_list[0] + "_" + file_list[i] + ".json")
        hit1_list.append(data["1"])
        hit5_list.append(data["5"])
        hit10_list.append(data["10"])
        hit15_list.append(data["15"])
        hit30_list.append(data["30"])
    
    obtain_multi_platforms(hit5_list, 2)
    obtain_multi_platforms(hit5_list, 3)
    obtain_multi_platforms(hit5_list, 4)
    obtain_multi_platforms(hit5_list, 5)
    
    obtain_multi_platforms(hit10_list, 2)
    obtain_multi_platforms(hit10_list, 3)
    obtain_multi_platforms(hit10_list, 4)
    obtain_multi_platforms(hit10_list, 5)

    obtain_multi_platforms(hit15_list, 2)
    obtain_multi_platforms(hit15_list, 3)
    obtain_multi_platforms(hit15_list, 4)
    obtain_multi_platforms(hit15_list, 5)

    obtain_multi_platforms(hit30_list, 2)
    obtain_multi_platforms(hit30_list, 3)
    obtain_multi_platforms(hit30_list, 4)
    obtain_multi_platforms(hit30_list, 5)
    pass