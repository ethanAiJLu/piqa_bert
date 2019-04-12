# piqa_bert
这是一个快速问答模型，基于2018年EMNLP发布的一篇论文和bert模型
论文地址 ： https://arxiv.org/abs/1804.07726
优点：在语义理解的基础上保持文档和问题之间的独立性，context和question单独编码，与传统的做法相比。速度上有了巨大的提提升，传统的做法是将context和questionyiqi
喂入一个神经网络模型中，基于question和context之间的attention机制来寻找到给定问题在文中的答案的起始位置，但是这样做非常缓慢，毕竟这样做的话对于一个问题来说，需要对每一篇文章
都做一次前向传播，计算量巨大，于是我参考了piqa论文中的模型，将context和question在模型中的前向传播独立开来，将context和question都单独编码为一个向量，预测的时候只需要
将输入的question前向传播得到向量，再将向量和已经编码号保存在本地的context的向量进行预测，就可以快速得到想要的答案，计算量和传统的方法比，不是一个数量级的，但是这样的做法也是
有一定的缺点的，缺点就是训练时候的模型非常大，因为context和question各自需要一个语义编码的模型，最终预测时候也需要一个预测模型，当然这里的预测模型可以使用一个简单
高效的模型，比如点积，欧式距离等，但是即便这样，模型相比传统的方法来说也比较大，而且模型的预测性能相比与传统的方法还有一定的差距，piqa这篇论文中的在squad1.1的数据集上
使用的模型的性能f1值只达到了62.7（lstm+sa+elmo）,本模型使用了最近大热的bert模型作为编码器，进行了测试，目前已经在一个1万个问题的question的语料上进行训练和测试，
在训练集上的效果已经超越了piqa论文中的效果，但是对于整个squad1.1数据集。由于笔者设备有限，暂时没有训练出来。
#环境要求
torch:0.4.1 
numpy：1.16.2 
pytorch_pretrained_bert
faiss
#训练命令：
python main.py --mode train --start-steps 0
#编码命令
python main.py baseline --cuda --mode embed_context --load_dir $OUTPUT_DIR/save/XXXX/model.pt --test_path   $SQUAD_DEV_CONTEXT_PATH --context_emb_dir $CONTEXT_EMB_DIR
python main.py baseline --cuda --mode embed_question --load_dir $OUTPUT_DIR/save/XXXX/model.pt --test_path $SQUAD_DEV_QUESTION_PATH --question_emb_dir $QUESTION_EMB_DIR
#测试命令
  训练集的测试命令在训练过程中会自动输出，测试集合的训练命令如下,注意把地址替换为你自己的地址
export SQUAD_DEV_PATH="/home/ethony/data/squad_data/dev-v1.1.json"
export CONTEXT_EMB="/home/ethony/data/my_piqa/emb/context_emb"
export QUESTION_EMB="/home/ethony/data/my_piqa/emb/question_emb"
export PRED_DIR="/home/ethony/data/my_piqa/pred/pred.json"
python merge.py $SQUAD_DEV_PATH $CONTEXT_EMB $QUESTION_EMB $PRED_DIR
python evaluate.py $SQUAD_DEV_PATH $PRED_DIR
