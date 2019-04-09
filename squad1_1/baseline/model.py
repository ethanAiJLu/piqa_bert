import torch
from torch import nn
from pytorch_pretrained_bert import BertModel
import numpy as np
import time

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
class contextBoundary(nn.Module):
    def __init__(self, hidden_size, dropout,bert_train=False):
        super(contextBoundary, self).__init__()
        self.dropout=torch.nn.Dropout(p=dropout)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to("cuda")
        self.linear_mdoel = torch.nn.Linear(768,hidden_size).to("cuda")
        self.bert_train = bert_train
        self.softmax = nn.Softmax(dim=2)
        self.highway = Highway(hidden_size,dropout)
        self.time_dict = {
            "bert_time":0,
            "linear_highway_time":0,
            "predict_time":0
        }
    def forward(self, x,mask):
        # print("进context boundary的时候，数据的shape为：",x.shape)
        time0 = time.time()
        if(self.bert_train):
            x,_=self.bert_model(x)
        else :
            self.bert_model.eval()
            with torch.no_grad():
                x, _ = self.bert_model(x)
        assert len(x) == 12
        x=x[-1]
        time1=time.time()
        # print("经过了bert模型之后的shape为:",x.shape)
        x=self.linear_mdoel(x)
        time2=time.time()
        self.time_dict["bert_time"]+=(time1-time0)
        self.time_dict["linear_highway_time"]+=(time2-time1)
        # x=self.softmax(x)
        # x=self.highway(x)
        # print("经过了linear层之后的shape为:",x.shape)
        return {"dense":x}
class questionBoundary(contextBoundary):
    def __init__(self,hidden_size,drop_out,max_pool,bert_train=False):
        super(questionBoundary, self).__init__(hidden_size,drop_out,bert_train)
        self.max_pool = max_pool
    def forward(self, x,mask):
        d=super().forward(x,mask)
        if(self.max_pool):
            dense=d["dense"].max(1)[0]
        else:
            dense=d["dense"][:,0,:]
        return {"dense":dense}

class Model(nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout,
                 max_pool=False,
                 ):
        super(Model, self).__init__()
        self.context_start = contextBoundary(hidden_size, dropout)
        self.context_end = contextBoundary( hidden_size, dropout, )
        self.question_start = questionBoundary( hidden_size, dropout, max_pool=max_pool)
        self.question_end = questionBoundary(hidden_size, dropout, max_pool=max_pool)
        self.softmax = nn.Softmax(dim=1)
        self.context_batch_num = 0
        self.question_batch_num=0
        self.max_ans_len=7
    def save_tensor_to_local(self,tensor,local_file):
        v_cpu = tensor.cpu()
        np.savetxt(local_file,v_cpu.detach().numpy())
    def forward(self,context_input_ids,context_input_mask,question_input_ids, question_input_mask):
        # print("喂给模型的context数据shape为:",context_input_ids.shape,"   ",context_input_mask.shape)
        # print("喂给模型的question数据的shape为：",question_input_ids.shape,"    ",question_input_mask.shape)
        # print("喂给model的数据的shape为：",question_word_idxs.shape)
        # q = self.question_embedding(question_input_ids, question_input_mask,question_segment_id)
        # x = self.context_embedding(context_input_ids,context_input_mask,context_segment_ids)
        qd1 = self.question_start(question_input_ids,question_input_mask)
        qd2 = self.question_end(question_input_ids,question_input_mask)
        q1 = qd1['dense']
        q2 = qd2['dense']
        # print("question经过了boundary模型之后的shape为：",q1.shape,"   ",q2.shape)
        mx = context_input_mask
        hd1 = self.context_start(context_input_ids,context_input_mask)
        hd2 = self.context_end(context_input_ids,context_input_mask)
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