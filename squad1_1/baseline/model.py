import torch
from torch import nn
from pytorch_pretrained_bert import BertModel
import numpy as np
import time
import json
import copy

class Highway(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Highway, self).__init__()
        self.input_linear = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.gate_linear = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_):
        input_ = self.dropout(input_)
        output = self.relu(self.input_linear(input_))
        gate = self.sigmoid(self.gate_linear(input_))
        output = input_ * gate + output * (1.0 - gate)
        return output
class BERTembed(nn.Module):
    def __init__(self,do_train=False):
        super(BERTembed, self).__init__()
        self.do_train=do_train
        self.bert_model=BertModel.from_pretrained("bert-base-uncased").to("cuda")
        self.cashed_context_dict = {}
        self.cashed_question_dict={}
        if(not do_train):
            self.cashed_context_np = np.zeros((10000, 512, 768))
            self.cashed_question_np = np.zeros((10000, 200, 768))
        self.cashed_context_num=0
        self.cashed_question_num=0
        self.cash_index=0
    def save_cash_to_local(self,save_dir = "/home/ethony/workstation/python_workstation/my_piqa/squad1_1/data/dict/"):
        np.save(save_dir+"context_matrix.data",self.cashed_context_np)
        np.save(save_dir+"question_matrix.data",self.cashed_question_np)
        with open(save_dir+"context_emb.json","w") as context_file:
            json.dump(self.cashed_context_dict,context_file)
        with open(save_dir+"question_emb.json","r") as question_file:
            json.dump(self.cashed_question_dict,question_file)
    def load_cashed_dict(self,save_dir = "/home/ethony/workstation/python_workstation/my_piqa/squad1_1/data/dict/"):
        self.cashed_context_np = np.load(save_dir + "context_matrix.data")
        self.cashed_question_np=np.load(save_dir + "question_matrix.data")
        with open(save_dir+"question_emb.json","r") as question_file:
            self.cashed_question_dict = json.load(question_file)
        with open(save_dir+"context_emb.json","w") as context_file:
            self.cashed_context_dict = json.load(self.context_file)
    def forward(self, input_ids,input_masks,ids,is_context):
        result = None
        if(not self.do_train):

            self.bert_model.eval()
            with torch.no_grad():
                if(is_context):
                    all_have = True
                    for id in ids:
                        if (not id in self.cashed_context_dict):
                            all_have = False
                    if(all_have):#如果需要的batch数据全部都已经有了的话
                        out = np.zeros((len(ids),512,768))
                        for i,col_index in enumerate([self.cashed_context_dict[x] for x in ids]):
                            out[i]=self.cashed_context_np[col_index]
                        result=torch.Tensor(out).cuda(0)
                    else:#如果输入的batch数据并不是全部都有的话
                        bert_out, _ = self.bert_model(input_ids, attention_mask=input_masks)
                        bert_out = bert_out[-1]
                        # assert bert_out.size()[0] == 768, "bert返回的不是768维度的数据，数据切分有问题"
                        for i,id in enumerate(ids):
                            if(id not in self.cashed_context_dict):
                                tmp_arr = np.array(bert_out.cpu().detach().numpy()[i])
                                print(tmp_arr.shape,"   ",type(tmp_arr))
                                # print(tmp_arr.shape,"    ",tmp_arr)
                                self.cashed_context_np[self.cashed_context_num] = tmp_arr # 将数据添加到缓存矩阵中
                                self.cashed_context_dict[id]=self.cashed_context_num#将矩阵的行与数据id的映射关系添加到字典中
                                self.cashed_context_num += 1
                        result = bert_out
                else:
                    all_have = True
                    for id in ids:
                        if (not id in self.cashed_question_dict):
                            all_have = False
                    if (all_have):  # 如果需要的batch数据全部都已经有了的话
                        out = np.zeros((len(ids), 512, 768))
                        for i, col_index in enumerate([self.cashed_question_dict[x] for x in ids]):
                            out[i] = self.cashed_question_np[col_index]
                        result = torch.Tensor(out).cuda(0)
                    else:#如果输入的batch数据并不是全部都有的话
                        bert_out, _ = self.bert_model(input_ids, attention_mask=input_masks)
                        bert_out = bert_out[-1]
                        # assert bert_out.size()[0] == 768, "bert返回的不是768维度的数据，数据切分有问题"
                        for i,id in enumerate(ids):
                            if(id not in self.cashed_question_dict):
                                tmp_arr = np.array(bert_out.cpu().detach().numpy()[i])
                                # print(tmp_arr.shape,"   ",type(tmp_arr))
                                # print(self.cashed_question_np.shape)
                                # print(tmp_arr.shape,"    ",tmp_arr)
                                self.cashed_question_np[self.cashed_question_num] = tmp_arr # 将数据添加到缓存矩阵中
                                self.cashed_question_dict[id]=self.cashed_question_num#将矩阵的行与数据id的映射关系添加到字典中
                                self.cashed_question_num += 1
                        result = bert_out

        elif(self.do_train):
            bert_out, _ = self.bert_model(input_ids,attention_mask=input_masks)
            result = bert_out[-1]
        return result
class contextBoundary(nn.Module):
    def __init__(self, hidden_size, dropout,bert_embed_model):
        super(contextBoundary, self).__init__()
        self.dropout=torch.nn.Dropout(p=dropout)
        self.embed_model = copy.copy(bert_embed_model)
        self.linear_mdoel = torch.nn.Linear(768,hidden_size).to("cuda")
        # self.softmax = nn.Softmax(dim=2)
        # self.highway = Highway(hidden_size,dropout)
        self.time_dict = {
            "bert_time":0,
            "linear_highway_time":0,
            "predict_time":0
        }
    def forward(self,x,mask,ids,is_context):
        # print("进context boundary的时候，数据的shape为：",x.shape)
        time0 = time.time()
        out = self.embed_model(x,mask,ids,is_context)
        time1=time.time()
        # print("经过了bert模型之后的shape为:",x.shape)
        out=self.linear_mdoel(out)
        time2=time.time()
        self.time_dict["bert_time"]+=(time1-time0)
        self.time_dict["linear_highway_time"]+=(time2-time1)
        # x=self.softmax(x)
        # x=self.highway(x)
        # print("经过了linear层之后的shape为:",x.shape)
        return {"dense":out}
class questionBoundary(contextBoundary):
    def __init__(self,hidden_size,drop_out,bert_embed_model):
        super(questionBoundary, self).__init__(hidden_size,drop_out,bert_embed_model)
    def forward(self, x,mask,ids,is_context=False):
        d=super().forward(x,mask,ids,is_context=False)
        # if(self.max_pool):
        #     dense=d["dense"].max(1)[0]
        # else:
        #     dense=d["dense"][:,0,:]
        dense=d["dense"][:,0,:]
        return {"dense":dense}

class Model(nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout,
                 bert_do_train=False
                 ):
        super(Model, self).__init__()
        self.bert_embed_model = BERTembed(do_train=bert_do_train)
        self.bert_do_train=bert_do_train
        self.context_start = contextBoundary(hidden_size, dropout,self.bert_embed_model)
        self.context_end = contextBoundary( hidden_size, dropout,self.bert_embed_model )
        self.question_start = questionBoundary( hidden_size, dropout,self.bert_embed_model)
        self.question_end = questionBoundary(hidden_size, dropout,self.bert_embed_model)
        self.softmax = nn.Softmax(dim=1)
        self.context_batch_num = 0
        self.question_batch_num=0
        self.max_ans_len=7
    def save_tensor_to_local(self,tensor,local_file):
        v_cpu = tensor.cpu()
        np.savetxt(local_file,v_cpu.detach().numpy())
    def forward(self,context_input_ids,context_input_mask,question_input_ids, question_input_mask,c_ids,question_ids,save_bert_dict=False,load_bert_embed=False):
        # print("喂给模型的context数据shape为:",context_input_ids.shape,"   ",context_input_mask.shape)
        # print("喂给模型的question数据的shape为：",question_input_ids.shape,"    ",question_input_mask.shape)
        # print("喂给model的数据的shape为：",question_word_idxs.shape)
        # q = self.question_embedding(question_input_ids, question_input_mask,question_segment_id)
        # x = self.context_embedding(context_input_ids,context_input_mask,context_segment_ids)
        if(save_bert_dict):
            if(self.bert_do_train):
                raise Exception("bert参与训练的时候缓存机制不能开启")
            self.bert_embed_model.save_cash_to_local()
        if(load_bert_embed):
            if (self.bert_do_train):
                raise Exception("bert参与训练的时候缓存机制不能开启")
            self.bert_embed_model.load_cashed_dict()
        qd1 = self.question_start(question_input_ids,question_input_mask,question_ids,is_context=False)
        qd2 = self.question_end(question_input_ids,question_input_mask,question_ids,is_context=False)
        q1 = qd1['dense']
        q2 = qd2['dense']
        # print("question经过了boundary模型之后的shape为：",q1.shape,"   ",q2.shape)
        hd1 = self.context_start(context_input_ids,context_input_mask,c_ids,is_context=True)
        hd2 = self.context_end(context_input_ids,context_input_mask,c_ids,is_context=True)
        time_dict = {
            "bert_time": 0,
            "linear_highway_time": 0,
            "predict_time": 0
        }
        for key in self.question_start.time_dict.keys():
            time_dict[key]+=self.question_start.time_dict[key]
            time_dict[key]+=self.question_end.time_dict[key]
            time_dict[key]+=self.context_end.time_dict[key]
            time_dict[key]+=self.context_start.time_dict[key]

        time10=time.time()
        x1 = hd1['dense']
        x2 = hd2['dense']
        # print("context经过了boundary之后的shape为 : ",x1.shape,"   ",x2.shape)
        logits1 = torch.sum(x1 * q1.unsqueeze(1), 2)
        logits2 = torch.sum(x2 * q2.unsqueeze(1), 2)
        # print("context和question内积之后的shape是 ：",logits1.shape,"   ",logits2.shape)
        # if self.metric == 'l2':
        #     logits1 += -0.5 * (torch.sum(x1 * x1, 2) + torch.sum(q1 * q1, 1).unsqueeze(1))
        #     logits2 += -0.5 * (torch.sum(x2 * x2, 2) + torch.sum(q2 * q2, 1).unsqueeze(1))

        prob1 = self.softmax(logits1)
        prob2 = self.softmax(logits2)
        prob = prob1.unsqueeze(2) * prob2.unsqueeze(1)
        # print("prob shape is :",prob.shape)
        mask = (torch.ones(*prob.size()[1:]).triu() - torch.ones(*prob.size()[1:]).triu(self.max_ans_len)).to(
            prob.device)
        # print("mask shape is :",mask.shape)
        prob *= mask
        # print("prob * mask shape is :",prob.shape)
        _, yp1 = prob.max(2)[0].max(1)
        _, yp2 = prob.max(1)[0].max(1)
        time11=time.time()
        time_dict["predict_time"]+=(time11-time10)
        # print("finally the yp shape is ",yp1.shape,"    ",yp2.shape)
        return {'logits1': logits1,
                'logits2': logits2,
                'yp1': yp1,
                'yp2': yp2,
                'x1': x1,
                'x2': x2,
                'q1': q1,
                'q2': q2,
                "time_dict":time_dict}
    def get_context(self,context_input_ids,context_input_mask):
        hd1 = self.context_start(context_input_ids, context_input_mask)
        hd2 = self.context_end(context_input_ids, context_input_mask)
        x1 = hd1['dense']
        x2 = hd2['dense']
        out = []
        l = (context_input_mask>0).sum(1)
        for k,(lb,x1b,x2b) in enumerate(zip(l,x1,x2)):#循环batch个句子
            pos_list = []
            vec_list = []
            for i in range(lb):#循环一句话的每一个个备选答案
                for j in range(i, min(i + self.max_ans_len, lb)):#循环每一句话中的备选答案的编码
                    vec = torch.cat([x1b[i], x2b[j]], 0)
                    pos_list.append((i, j))
                    vec_list.append(vec)
            dense = torch.stack(vec_list, 0)#得到每一个备选答案的编码列表
            out.append((tuple(pos_list), dense))#备选答案的位置已经它的编码列表
        return tuple(out)
    def get_question(self, question_input_ids, question_input_mask):
        qd1 = self.question_start(question_input_ids, question_input_mask)
        qd2 = self.question_end(question_input_ids, question_input_mask)
        q1 = qd1['dense']
        q2 = qd2['dense']
        out = list(torch.cat([q1, q2], 1).unsqueeze(1))
        return out

class Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Loss, self).__init__()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, logits1, logits2, answer_word_starts, answer_word_ends, **kwargs):
        # answer_word_starts -= 1
        # answer_word_ends -= 1
        loss1 = self.cel(logits1, answer_word_starts)
        loss2 = self.cel(logits2, answer_word_ends)
        loss = loss1 + loss2
        return loss