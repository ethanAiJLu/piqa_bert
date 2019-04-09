import json
source_train_data_path = "/home/ethony/workstation/python_workstation/my_piqa/squad1_1/data/train-v1.1.json"
with open(source_train_data_path) as file:
    obj = json.load(file)
mini = {}
mini["data"] = []
mini["version"] = "1.1"
mini_file_path = "/home/ethony/workstation/python_workstation/my_piqa/squad1_1/data/train_mini_3.json"
print("将会分割15%左右的数据到新的小数据集中,切分了{0}个paragraph到新的小数据集中".format(int(0.15*len(obj["data"]))))
for i in range(int(0.15*len(obj["data"]))):
    mini["data"].append(obj["data"][i])
with open(mini_file_path,"w") as mini_file:
    json.dump(mini,mini_file)
print("切分成功，请到该目录下查看与使用新的小数据集:{0}".format(mini_file_path))

