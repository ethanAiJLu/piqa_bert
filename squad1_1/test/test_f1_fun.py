import torch
from collections import Counter
import numpy as np

a=torch.Tensor([[1,2,3,4,5,6],[1,2,3,7,8,9]])
pre_start = torch.Tensor([1,3])
pre_end = torch.Tensor([3,5])
label_start = torch.Tensor([1,4])
label_end = torch.Tensor([3,7])

a_ = np.array(a,dtype=int)
pre_start_ = np.array(pre_start,dtype=int)
pre_end_ = np.array(pre_end,dtype=int)
label_start_ = np.array(label_start,dtype=int)
label_end_ = np.array(label_end,dtype=int)
b = list(a_[0])
print(b)

common = Counter(b[pre_start_[0]:pre_end_[0]])&Counter(b[label_start_[0]:label_end_[0]])
print(common)