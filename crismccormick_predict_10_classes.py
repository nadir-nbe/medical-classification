import flask
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from typing import List
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import pipeline


translator_tokenizer_fr_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
translator_model_fr_en = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-fr-en")


def translate_fr_en(text):
    batch = translator_tokenizer_fr_en.prepare_translation_batch(src_texts=[text])
    gen = translator_model_fr_en.generate(**batch)  # for forward pass: model(**batch)
    translated = translator_tokenizer_fr_en.batch_decode(gen, skip_special_tokens=True)
    return str(translated[0])


net = BertForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=10,  # For our 20 newsgroups!
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
# net = nn.DataParallel(net)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_state_dict = torch.load('./Models/clinical_5_39_categories.dat', map_location=torch.device("cpu"))

net.load_state_dict(model_state_dict)
net.eval()

print("One")
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True,
                                          device=torch.device("cpu"))
# device = torch.device("cpu")
# tokenizer.to(device)
# print(tokenizer.vocab)
print("Two")
T = 500

# instantiate flask
app = flask.Flask(__name__)


def get_sentence_features(tokens: List[int], pad_seq_length: int, cls_token_id: int, sep_token_id: int):
    """
    Convert tokenized sentence in its embedding ids, segment ids and mask
    :param tokens:
        a tokenized sentence
    :param pad_seq_length:
        the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
    :return: embedding ids, segment ids and mask for the sentence
    """
    pad_seq_length = min(pad_seq_length, 511)

    tokens = tokens[:pad_seq_length]
    input_ids = [cls_token_id] + tokens + [sep_token_id] + [sep_token_id]
    sentence_length = len(input_ids)

    pad_seq_length += 3  ##Add Space for CLS + SEP + SEP token

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length. BERT: Pad to the right
    padding = [0] * (pad_seq_length - len(input_ids))
    input_ids += padding

    input_mask += padding
    return input_ids, input_mask


def tokinize(sentence):
    tokens_ids, attn_mask = get_sentence_features(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)), T, 5,
                                                  6)
    return tokens_ids, attn_mask


# define a predict function as an endpoint
@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {}
    params = flask.request.json
    print("I get : ", params)
    if (params == None):
        params = flask.request.args

    if (params != None):

        sentence = params['data'][0]

        answer = []
        for sen in params['data']:
            tokens_ids, attn_mask = tokinize(sen['text'])  # tokinize(translate_fr_en(sen['text']))#
            with torch.no_grad():
                logits = net(torch.tensor(tokens_ids).unsqueeze(0), torch.tensor(attn_mask).unsqueeze(0))
                m = nn.Softmax(dim=1)
                result_list = torch.tensor(m(logits[0])).cpu().numpy()
                result_list = result_list.flatten()
            answer.append({"id": sen['id'],
                           "classes": {"Gastroenterology": str(result_list[0]), "Neurology": str(result_list[1]),
                                       "Orthopedic": str(result_list[2]), "Radiology": str(result_list[3]),
                                       "Urology": str(result_list[4]), "Obstetrics / Gynecology": str(result_list[5]),
                                       "Discharge Summary": str(result_list[6]),
                                       "ENT - Otolaryngology": str(result_list[7]),
                                       "Hematology - Oncology": str(result_list[8]),
                                       "Neurosurgery": str(result_list[9])}})

        # toxique	toxique_severe	  obscene	menace   	insulte   	haine_raciale
        data["prediction"] = answer
        # data["success"] = True

    # return a response in json format
    return data



nlp = pipeline('question-answering', model='fmikaelian/camembert-base-squad', tokenizer='fmikaelian/camembert-base-squad')

@app.route("/question", methods=["GET", "POST"])
def question(charset='utf-8'):
    data = {}
    params = flask.request.json
    if (params == None):
        params = flask.request.args

    if (params != None):
        answer = []
        for sen in params['data']:
            response = nlp({
                'question': sen['question'],
                'context': sen['context']
            })
            answer.append({"id":sen['id'],"answer":response['answer'],"score":response['score']})
    data["prediction"] = answer
    print("data",data)
    return data
# start the flask app, allow remote connections
app.run(host='127.0.0.1')