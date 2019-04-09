from pytorch_pretrained_bert import BertTokenizer
import torch
import random
from torch.utils.data import Sampler
import base
def _get_shape(nested_list, depth):
    if depth > 0:
        return (len(nested_list),) + tuple(map(max, zip(*[_get_shape(each, depth - 1) for each in nested_list])))
    return ()
class Processor():
    def __init__(self):
        self.bert_tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
        self.keys = {
            "context_words_idx",
            "question_words_idx",
            "answer_word_starts",
            "answer_word_ends"
        }
        self.depth={
            "context_words_idx":1,
            "question_words_idx":1,
            "answer_word_starts":1,
            "answer_word_ends":1
        }
    def _word_toknize(self,context):
        words = self.bert_tokenizer.tokenize(context)
        return words
    def _words_to_id(self,tokens):
        tokens = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        return tokens
    def preprocess(self,example):
        prepro_example = {"idx":example["idx"]}
        if("context" in example):
            context = example["context"]
            context_words_tokens = self._word_toknize(context.lower())
            context_words_idx = self._words_to_id(context_words_tokens)
            prepro_example["context_words_idx"] = context_words_idx
        if("question" in example):
            question = example["question"]
            question_words_tokens=self._word_toknize(question)
            question_words_idx = self._words_to_id(question_words_tokens)
            prepro_example["question_words_idx"]=question_words_idx
        def get_index(answer_word_tokens):
            start = -1
            while (start >= -1 and start <= len(context_words_tokens)):
                start = context_words_tokens.index(answer_word_tokens[0], start + 1)
                end = context_words_tokens.index(answer_word_tokens[-1], start)
                if (end - start) < (len(answer_word_tokens)) + 3:
                    return (start, end)
            raise Exception("没有寻找到答案")
        if("answers" in example):
            answer_word_starts, answer_word_ends = [], []
            for answer_word in example["answers"]:
                answer_word_tokens = self.bert_tokenizer.tokenize(answer_word)
                answer_word_idx = self.bert_tokenizer.convert_tokens_to_ids(answer_word_tokens)
                answer_word_start, answer_word_end = get_index(answer_word_idx)
                answer_word_starts.append(answer_word_start)
                answer_word_ends.append(answer_word_end)
            prepro_example['answer_word_starts'] = answer_word_starts
            prepro_example['answer_word_ends'] = answer_word_ends
    #设置如何将多个样本数据拼接为一个batch
    def collate(self,examples):
        tensors={}
        for key in self.keys:
            if(key not in examples[0]):
                continue
            val=tuple(example[key] for example in examples)
            depth=self.depth[key]+1
            shape=_get_shape(val,depth)
            tensor=torch.zeros(shape,dtype=torch.int64)
            _fill_tensor(tensor,val)
            tensors[key]=tensor
        return tensors

def _fill_tensor(tensor, nested_list):
    if tensor.dim() == 1:
        tensor[:len(nested_list)] = torch.tensor(nested_list)
    elif tensor.dim() == 2:
        for i, each in enumerate(nested_list):
            tensor[i, :len(each)] = torch.tensor(each)
    elif tensor.dim() == 3:
        for i1, each1 in enumerate(nested_list):
            for i2, each2 in enumerate(each1):
                tensor[i1, i2, :len(each2)] = torch.tensor(each2)
    else:
        for tensor_child, nested_list_child in zip(tensor, nested_list):
            _fill_tensor(tensor_child, nested_list_child)
class Sampler(Sampler):
    def __init__(self, dataset, data_type, max_context_size=None, max_question_size=None, bucket=False, shuffle=False,
                 **kwargs):
        self.dataset=dataset
        self.data_type = data_type
        if data_type == 'dev' or data_type == 'test':
            max_context_size = None
            max_question_size = None
            self.shuffle = False

        self.max_context_size = max_context_size
        self.max_question_size = max_question_size
        self.shuffle = shuffle
        self.bucket = bucket

        idxs = tuple(idx for idx in range(len(dataset))
                     if (max_context_size is None or len(dataset[idx]['context_spans']) <= max_context_size) and
                     (max_question_size is None or len(dataset[idx]['question_spans']) <= max_question_size))

        if shuffle:
            idxs = random.sample(idxs, len(idxs))

        if bucket:
            if 'context_spans' in dataset[0]:
                idxs = sorted(idxs, key=lambda idx: len(dataset[idx]['context_spans']))
            else:
                assert 'question_spans' in dataset[0]
                idxs = sorted(idxs, key=lambda idx: len(dataset[idx]['question_spans']))
        self._idxs = idxs

    def __iter__(self):
        return iter(self._idxs)

    def __len__(self):
        return len(self._idxs)
