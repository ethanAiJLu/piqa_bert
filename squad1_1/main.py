from tqdm import tqdm
import torch
from torch import nn
from baseline import Loader
from baseline import Model
from baseline import Loss
from baseline import processor
from baseline import FileInterface
# from base import ArgumentParser
import argparse
import time
from pytorch_pretrained_bert import BertTokenizer,BertModel
from collections import Counter
import os
import numpy as np
import json

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sep_id = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize("[SEP]"))[0]
def get_sep_index(tensor):
    list_tensor = list(tensor)
    index = list_tensor.index(sep_id)
    index2 = list_tensor.index(sep_id, index + 1)
    out = [tensor[1:index], tensor[index + 1:index2], index, index2]
    return out
def get_context(tensor):
    cat_len = 512 - (tensor[3] - tensor[2]) + 1
    cat_tensor = torch.zeros(cat_len).long()
    out = torch.cat((tensor[1], cat_tensor), 0)
    return out
def get_question(tensor):
    cat_len = 200 - tensor[2] + 1
    cat_tensor = torch.zeros(cat_len).long()
    out = torch.cat((tensor[0], cat_tensor), 0)
    return out
def handle_batch(inputs_id):
    context_sep = list(map(get_sep_index, inputs_id))
    context_idx = list(map(get_context, context_sep))
    question_idx = list(map(get_question, context_sep))
    context_idx = torch.stack(context_idx, 0).to("cuda")
    context_mask = context_idx > 0
    question_idx = torch.stack(question_idx, 0).to("cuda")
    question_mask = question_idx > 0
    return context_idx, context_mask, question_idx, question_mask
def strcu_context_emb_local(context_bert_dir,device="cuda",batch_size=96):
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

def train(epochs,train_model_dir,train_mode,dev_mode,device="cuda",device_id=[0],train_model_start_step=0,batch_size=36,n_gpu=1):
    loader = Loader(batch_size=batch_size,train_mode=train_mode,dev_mode=dev_mode)
    data_out = loader.get_data_loader()
    train_loader = data_out["train_data_loader"]
    dev_loader=data_out["dev_data_loader"]

    def _f1_score(batch_data, pre_start, pre_end, label_start, label_end):
        f1_results = []
        batch_data_np = np.array(batch_data, dtype=int)
        pre_start_np = np.array(pre_start, dtype=int)
        pre_end_np = np.array(pre_end, dtype=int)
        label_start_np = np.array(label_start, dtype=int)
        label_end_np = np.array(label_end, dtype=int)
        for i in range(batch_data.size()[0]):
            context = list(batch_data_np[i])
            comman = Counter(context[pre_start_np[i]:pre_end_np[i]+1]) & Counter(
                context[label_start_np[i]:label_end_np[i]+1])
            num_same = sum(comman.values())
            if (num_same == 0):
                f1_results.append(0)
            else:
                precision = 1.0 * num_same / (pre_end_np[i]+1 - pre_start_np[i])
                recall = 1.0 * num_same / (label_end_np[i]+1 - label_start_np[i])
                f1_score = (2 * precision * recall) / (precision + recall)
                f1_results.append(f1_score)
        return {"f1_scores": f1_results}

    time_dict = {
        "handle_time":0,
        "forward_time":0,
        "backward_time":0
    }
    model = Model(hidden_size=512,dropout=0.5,max_pool=False).cuda(0)
    if(n_gpu>1):
        model = nn.DataParallel(model,device_ids = device_id)
        if (train_model_start_step > 0):
            load_model_dir = train_model_dir + "/piqa_{0}.pkl".format(train_model_start_step)
            if (os.path.exists(load_model_dir)):
                model.load_state_dict(torch.load(load_model_dir))
            else:
                raise FileNotFoundError
        model=model.module
    else:
        if (train_model_start_step > 0):
            load_model_dir = train_model_dir + "/piqa_{0}.pkl".format(train_model_start_step)
            if (os.path.exists(load_model_dir)):
                model.load_state_dict(torch.load(load_model_dir))
            else:
                raise FileNotFoundError
    loss_model=Loss().to("cuda")
    optimizer = torch.optim.Adam(p for p in model.parameters() if p.requires_grad)
    steps = train_model_start_step
    f1_scores = []
    scores_data = {}
    dev_scores = {}
    model_time_dict={
        "bert_time": 0,
        "linear_highway_time": 0,
        "predict_time": 0
    }
    print("training......")
    model.train()
    for epoch_idx in range(epochs):
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            time0 = time.time()
            # if n_gpu == 1:
            #     batch = tuple(t for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch["tensor_data"]
            c_ids = batch["c_ids"]
            question_ids = batch["question_id"]
            context_idx, context_mask, question_idx, question_mask = handle_batch(input_ids)
            time1=time.time()
            model_output = model(context_idx,context_mask,question_idx, question_mask)
            train_loss=loss_model(model_output["logits1"], model_output["logits2"], start_positions.to("cuda"), end_positions.to("cuda"))
            time2 = time.time()
            train_batch_f1_score = np.mean(_f1_score(input_ids, model_output["yp1"], model_output["yp2"], start_positions, end_positions)["f1_scores"])
            model_time_dict["bert_time"]+=model_output["time_dict"]["bert_time"]
            model_time_dict["linear_highway_time"] += model_output["time_dict"]["linear_highway_time"]
            model_time_dict["predict_time"] += model_output["time_dict"]["predict_time"]
            f1_scores.append(train_batch_f1_score)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            time3=time.time()
            time_dict["handle_time"]+=(time1-time0)
            time_dict["forward_time"]+=(time2-time1)
            time_dict["backward_time"]+=(time3-time2)
            steps+=1
            if(steps%10==0):
                print("*********************************************************************")
                print("切分数据的时间占比为:{0}".format(time_dict["handle_time"] / (
                            time_dict["handle_time"] + time_dict["forward_time"] + time_dict["backward_time"])))
                print("前向传播的时间占比为:{0}".format(time_dict["forward_time"] / (
                            time_dict["handle_time"] + time_dict["forward_time"] + time_dict["backward_time"])))
                print("反向传播的时间占比为：{0}".format(time_dict["backward_time"] / (
                            time_dict["handle_time"] + time_dict["forward_time"] + time_dict["backward_time"])))
                print("在前向传播的过程中的时间占比如下：")

                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("bert前向传播时间占比为{0}".format(model_time_dict["bert_time"] / (
                            model_time_dict["predict_time"] + model_time_dict["bert_time"] + model_time_dict["linear_highway_time"])))
                print("linear前向传播时间占比为{0}".format(model_time_dict["linear_highway_time"] / (
                        model_time_dict["predict_time"] + model_time_dict["bert_time"] + model_time_dict["linear_highway_time"])))
                print("用向量预测起始位置时间占比为{0}".format(model_time_dict["predict_time"] / (
                        model_time_dict["predict_time"] + model_time_dict["bert_time"] + model_time_dict["linear_highway_time"])))
                print("--------------------------------------------------------------------------")
                print("#######################################################################")

            if(steps%100==0):
                print("run to {0}   ".format(step), "the train_loss(CrossEntropyLoss) : {0}".format(train_loss))
                # print("在这个batch上的max_f1_score值为:",np.max(train_batch_f1_scores))
                mean_scores = np.mean(f1_scores)
                f1_scores.clear()
                print("在过去的100个batch上的平均f1值为:",mean_scores)
                scores_data[steps] = mean_scores
            if(steps%500==0):
                model_save_dir = train_model_dir+"/piqa_{0}.pkl".format(steps)
                f1_json_dir = train_model_dir+"/piqa_{0}".format(steps)+"f1_scores_data.json"
                print("第{0}步的模型已经保存到{1}".format(steps,model_save_dir))
                torch.save(model.state_dict(),model_save_dir)
                with open(f1_json_dir,"w") as f1:
                    json.dump(scores_data,f1)
                    print("f1值数据已经保存到{0}".format(f1_json_dir))
            if (steps % 1000 == 0):
                dev_f1_scores = []
                with torch.no_grad():
                    model.eval()
                    for dev_step, dev_batch in enumerate(tqdm(dev_loader, desc="Iteration")):
                        if n_gpu == 1:
                            dev_batch = tuple(t for t in dev_batch["tensor_data"])
                        dev_input_ids, dev_input_mask, dev_segment_ids, dev_start_positions, dev_end_positions = dev_batch
                        dev_context_idx, dev_context_mask, dev_question_idx, dev_question_mask = handle_batch(
                            dev_input_ids)
                        dev_model_output = model(dev_context_idx, dev_context_mask, dev_question_idx, dev_question_mask)
                        dev_batch_f1_score = np.mean(
                            _f1_score(dev_input_ids, dev_model_output["yp1"], dev_model_output["yp2"],
                                      dev_start_positions, dev_end_positions)["f1_scores"])
                        dev_f1_scores.append(dev_batch_f1_score)
                    dev_scores[steps] = np.mean(dev_f1_scores)
                    with open(train_model_dir+"/"+str(steps)+"_dev_f1_scores.json","w") as dev_f1_file:
                        json.dump(dev_scores,dev_f1_file)
                        print("测试集的f1值数据已经保存到{0}".format(train_model_dir+"/"+str(steps)+"/dev_f1_scores.json"))
                print("***********************************************************")
                print("训练到第{0}步的时候模型在测试集上的平均f1值为{1}".format(steps, np.mean(dev_f1_scores)))
                print("############################################################")
def embed(train_model_dir,embed_dir,start_steps,has_metadata,emb_type,embed_mode="embed_context",n_gpu=1,archive=False):
    processor_obj = processor()
    file_interface = FileInterface()
    if(embed_mode=="embed_context"):
        model = Model(hidden_size=512, dropout=0.5, max_pool=False).cuda(0)
        load_model_dir = train_model_dir + "/piqa_{0}.pkl".format(start_steps)
        if (os.path.exists(load_model_dir)):
            model.load_state_dict(torch.load(load_model_dir))
        else:
            raise FileNotFoundError
        with torch.no_grad():
            model.eval()
        dev_loader = Loader()
        dev_data_loader = dev_loader.get_dev_loader()["dev_data_loader"]
        for dev_step, dev_batch in enumerate(tqdm(dev_data_loader, desc="Iteration")):
            c_ids = [c for c in dev_batch["c_ids"]]
            dev_batch = tuple(t for t in dev_batch["tensor_data"])
            dev_input_ids, dev_input_mask, dev_segment_ids, dev_start_positions, dev_end_positions = dev_batch
            dev_context_idx, dev_context_mask, dev_question_idx, dev_question_mask = handle_batch(
                dev_input_ids)
            output = model.get_context(dev_context_idx,dev_context_mask)#[(tuple(pos_list), dense),'''''']
            # print(c_ids)
            # print(dev_context_idx)
            # print(output)
            # print(c_ids[0], "    ",dev_context_idx[0],"   ",output[0])
            context_results = processor_obj.postprocess_context_batch(c_ids,dev_context_idx.cpu().detach().numpy(),output)
            for c_id,phrase,matrix,metadata in context_results:
                if(not has_metadata):
                    metaddata = None
                file_interface.context_emb(c_id,phrase,matrix,metadata=metadata,emb_type=emb_type,emb_dir=embed_dir)
    if(embed_mode=="embed_question"):
        model = Model(hidden_size=512, dropout=0.5, max_pool=False).cuda(0)
        load_model_dir = train_model_dir + "/piqa_{0}.pkl".format(start_steps)
        if (os.path.exists(load_model_dir)):
            model.load_state_dict(torch.load(load_model_dir))
        else:
            raise FileNotFoundError
        with torch.no_grad():
            model.eval()
        dev_loader = Loader()
        dev_data_loader = dev_loader.get_dev_loader()["dev_data_loader"]
        for dev_step, dev_batch in enumerate(tqdm(dev_data_loader, desc="Iteration")):
            question_ids = [q_id for q_id in dev_batch["question_id"]]
            dev_batch = tuple(t for t in dev_batch["tensor_data"])
            dev_input_ids, dev_input_mask, dev_segment_ids, dev_start_positions, dev_end_positions = dev_batch
            dev_context_idx, dev_context_mask, dev_question_idx, dev_question_mask = handle_batch(
                dev_input_ids)
            output = model.get_question(dev_question_idx, dev_question_mask)  # [(tuple(pos_list), dense),'''''']
            question_result = processor_obj.postprocess_question_batch(question_ids,
                                                                      output)
            for q_id,emb in question_result:
                file_interface.question_emb(q_id,emb,emb_type=emb_type,emb_dir=embed_dir)
    if(archive):
        processor_obj.archive(mode=embed_mode,context_emb="/home/ethony/data/my_piqa/emb/context_emb",question_emb="/home/ethony/data/my_piqa/emb/question_emb")
    def server():
        model = Model(hidden_size=512, dropout=0.5, max_pool=False).cuda(0)
        load_model_dir = train_model_dir + "/piqa_{0}.pkl".format(start_steps)
        if (os.path.exists(load_model_dir)):
            model.load_state_dict(torch.load(load_model_dir))
        else:
            raise FileNotFoundError
        pass
if __name__ == "__main__":
   parse = argparse.ArgumentParser(description="my piqa")
   parse.add_argument("--start-steps",type=int,default=0)
   parse.add_argument("--mode",type=str,default="train")
   args = parse.parse_args()
   if(args.mode=="train"):
      start_steps = args.start_steps
      train(epochs=1000,train_model_dir = "/home/ethony/data/my_piqa/trained_model",train_mode = "mini",dev_mode="mini",device="cuda",device_id = [0],train_model_start_step=start_steps)
   if(args.mode=="embed_context"):
      start_steps=args.start_steps
      embed(train_model_dir = "/home/ethony/data/my_piqa/trained_model",embed_dir="/home/ethony/data/my_piqa/emb/context_emb",start_steps=start_steps,has_metadata=True,emb_type="sparse",embed_mode="embed_context")
   if(args.mode=="embed_question"):
      start_steps=args.start_steps
      embed(train_model_dir = "/home/ethony/data/my_piqa/trained_model",embed_dir="/home/ethony/data/my_piqa/emb/question_emb",start_steps=start_steps,has_metadata=True,emb_type="sparse",embed_mode="embed_question")









