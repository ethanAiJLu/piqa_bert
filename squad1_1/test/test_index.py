import torch
import numpy as np
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
def get_index(tensor):
   index = list(tensor).index(3)
   out = [tensor[1:index],tensor[index+1:-1]]
   return out
def get_context(list_):
    return list_[0]
def get_question(list_):
    return list_[1]
a=torch.Tensor([[1,2,3,4,5,6,3],
   [1,2,3,4,5,6,3],
                [1,2,3,4,5,6,6]
   ])
b=list(map(get_index,a))
contexts = torch.Tensor(list(map(get_context,b)))
questions = list(map(get_question,b))
print(b)
print("###################")
print(len(contexts))
print(contexts)
print("!!!!!!!!!!!!!!!!!!!!!!!")
print(questions)
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# print(bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize("[SEP]")))
# print(bert_tokenizer.convert_ids_to_tokens([102]))