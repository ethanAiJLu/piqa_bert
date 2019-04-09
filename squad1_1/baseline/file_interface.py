#
# import json
# import nltk
# import torch
# import os
# from pytorch_pretrained_bert import BertTokenizer
# class FileInterface():
#     def __init__(self):
#         self.train_dir = "/home/ethony/data/squad_data/train-v1.1.json"
#         self.dev_dir = "/home/ethony/data/squad_data/dev-v1.1.json"
#         self.model_dir = "/home/ethony/data/my_piqa/trained_model/"
#         self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     def load_train(self):
#         return self._load_squad(self.train_dir)
#     def load_test(self):
#         return self._load_squad(self.dev_dir)
#     def _load_squad(self,squad_path):
#         with open(squad_path, 'r') as fp:
#             squad = json.load(fp)
#             examples = []
#             for article in squad['data']:
#                 for para_idx, paragraph in enumerate(article['paragraphs']):
#                     cid = '%s_%d' % (article['title'], para_idx)
#                     if 'context' in paragraph:
#                         context = paragraph['context'].lower()
#                         context_example = {'cid': cid, 'context': context}
#                         if (len(self.bert_tokenizer.tokenize(context)) > 512):
#                             continue
#                     else:
#                         context_example = {}
#
#                     if 'qas' in paragraph:
#                         for question_idx, qa in enumerate(paragraph['qas']):
#                             id_ = qa['id']
#                             qid = '%s_%d' % (cid, question_idx)
#                             question = qa['question'].lower()
#                             question_example = {'id': id_, 'qid': qid, 'question': question}
#                             if 'answers' in qa:
#                                 answers, answer_starts, answer_ends = [], [], []
#                                 for answer in qa['answers']:
#                                     answer_start = answer['answer_start']
#                                     answer_end = answer_start + len(answer['text'])
#                                     answers.append(answer['text'])
#                                     answer_starts.append(answer_start)
#                                     answer_ends.append(answer_end)
#                                 answer_example = {'answers': answers, 'answer_starts': answer_starts,
#                                                   'answer_ends': answer_ends}
#                                 question_example.update(answer_example)
#
#                             example = {'idx': len(examples)}
#                             example.update(context_example)
#                             example.update(question_example)
#                             examples.append(example)
#                             # if draft and len(examples) == 100:
#                             #     return examples
#                     else:
#                         example = {'idx': len(examples)}
#                         example.update(context_example)
#                         examples.append(example)
#                         # if draft and len(examples) == 100:
#                         #     return examples
#             return examples
#     def load_model(self,model,step = 0):
#         model_dir = self.model_dir+str(step)+"/model.pkl"
#         if(os.path.isfile(model_dir)):
#             model.load_state_dict(torch.load(model_dir))
#         else:
#             raise FileExistsError("模型文件不存在")
#     def save_model(self,step,model):
#         model_dir = self.model_dir+str(step)+"/model.pkl"
#         torch.save(model.state_dict(),model_dir)
#

import os
import scipy
import numpy as np
import json
class FileInterface(object):
    def __init__(self):
        pass
        # self._context_emb_dir="/home/ethony/data/my_piqa/emb/context_emb"
        # self._question_emb_dir="/home/ethony/data/my_piqa/emb/question_emb"
    def context_emb(self,id,phrase,emb,metadata,emb_type,emb_dir):
        if not  os.path.exists(emb_dir):
            os.mkdir(emb_dir)
        savez=scipy.sparse.save_npz if emb_type=="sparse" else np.savez_compressed
        emb_path = os.path.join(emb_dir,"{0}.npz".format(id))
        json_path=os.path.join(emb_dir,"{0}.json".format(id))
        if(os.path.exists(emb_path)):
            print("skipping {0};already exists")
        else:
            savez(emb_path,emb)

        if(os.path.exists(json_path)):
            print("skipping {0};already exists")
        else:
            with open(json_path,"w") as fp:
                json.dump(phrase,fp)
        if metadata is not None:
            metadata_path=os.path.join(emb_dir,"{0}.metadata".format(id))
            with open(metadata_path,"w") as fp:
                json.dump(metadata,fp)
    def question_emb(self,q_id,emb,emb_type,emb_dir):
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir)
        savez = scipy.sparse.save_npz if emb_type == 'sparse' else np.savez_compressed
        path = os.path.join(emb_dir, '%s.npz' % q_id)
        savez(path, emb)