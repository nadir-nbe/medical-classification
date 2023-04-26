import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class SentimentClassifier(nn.Module):

    def __init__(self, freeze_bert = True):
        super(SentimentClassifier, self).__init__()
        #Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        print("freeze_bert",freeze_bert)
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.layer_drop = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(768, 5)



    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        #print("seq, attn_masks" , seq,  attn_masks)
        #Feeding the input to BERT model to obtain contextualized representations

        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        #print("cont_reps, _ ", cont_reps[0])
        #Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]
        #print("cls_rep  ", cls_rep[0])
        #Feeding cls_rep to the classifier layer
        dropped = self.layer_drop(cls_rep)
        logits = self.cls_layer(dropped)
        #print("logits Z ",logits)
        #logits_numpy = logits.detach().cpu().numpy()
        #print("bef",logits_numpy)
        #logits[:, [1, 3]] += 3.0
        #logits = torch.from_numpy(logits_numpy).float().to('cuda')
        #print("af",logits)
        return logits
#32, 25, 768
#[32, 768]