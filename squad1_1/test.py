from baseline import Loader

loader = Loader()
dev_loader = loader.get_dev_loader()["dev_data_loader"]
for data in dev_loader:
    print(data["tensor_data"])
    print(data["c_ids"])
    break