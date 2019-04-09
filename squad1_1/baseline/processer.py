import numpy
from pytorch_pretrained_bert import BertTokenizer
from scipy.sparse import csc_matrix
import shutil
class processor():
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._emb_type="sparse"
    def _get_pred(self,context,yp1,yp2):
        return self.bert_tokenizer.convert_ids_to_tokens(context[yp1:yp2+1])
    def _get_spans(self,example,yp1,yp2):
        start=0
        end=0
        context = self.bert_tokenizer.convert_ids_to_tokens(example)
        for i in range(yp1):
            start+=len(context[i])
        start+1
        for i in range(yp2+1):
            end+=len(context[i])
        return (start,end)
    def postprocess_context(self,c_id,example,output):
        pos_tuple,dense = output
        out=dense.cpu().detach().numpy()
        phrase=tuple(self._get_pred(example,yp1,yp2) for yp1,yp2 in pos_tuple)
        if self._emb_type=='sparse':
            out=csc_matrix(out)
        metadata = {
            "context":self.bert_tokenizer.convert_ids_to_tokens(example),
            "answer_spans":tuple(self._get_spans(example,yp1,yp2) for yp1,yp2 in pos_tuple)
        }
        return c_id,phrase,out,metadata
    def postprocess_context_batch(self,c_ids,dataset,context_output):
        result = tuple([self.postprocess_context(c_ids[i],dataset[i],context_output[i]) for i in range(len(context_output))])
        return result
    def postprocess_question(self,question_id,question_output):
        dense=question_output
        out=dense.cpu().detach().numpy()
        if(self._emb_type=="sparse"):
            out=csc_matrix(out)
        return question_id,out
    def postprocess_question_batch(self,question_ids,question_output):
        result=tuple([self.postprocess_question(question_ids[i],question_output[i]) for i in range(len(question_ids))])
        return result
    def archive(self,mode,context_emb,question_emb):
        if mode == 'embed' or mode == 'embed_context':
            shutil.make_archive(context_emb, 'zip', context_emb)
            shutil.rmtree(context_emb)

        if mode == 'embed' or mode == 'embed_question':
            shutil.make_archive(question_emb, 'zip',question_emb)
            shutil.rmtree(question_emb)