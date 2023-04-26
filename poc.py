from transformers import AutoTokenizer, AutoModel,BertForMaskedLM,BertTokenizer,BertForMaskedLM
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead
import json


translator_tokenizer_fr_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
translator_model_fr_en = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

def translate_fr_en(text):
    batch = translator_tokenizer_fr_en.prepare_translation_batch(src_texts=[text])
    gen = translator_model_fr_en.generate(**batch)  # for forward pass: model(**batch)
    translated = translator_tokenizer_fr_en.batch_decode(gen, skip_special_tokens=True)
    return str(translated[0])

translator_tokenizer_en_fr = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
translator_model_en_fr = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

def translate_en_fr(text):
    batch = translator_tokenizer_en_fr.prepare_translation_batch(src_texts=[text])
    gen = translator_model_en_fr.generate(**batch)  # for forward pass: model(**batch)
    translated = translator_tokenizer_en_fr.batch_decode(gen, skip_special_tokens=True)
    return str(translated[0])



tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
model = BertForMaskedLM.from_pretrained("allenai/scibert_scivocab_uncased")
print (len(tokenizer))

fill_mask_original_scibert = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)
phrase = "Dans quel cas le médicament LEXOMIL est-il prescrit ? Ce médicament est un anxiolytique (tranquillisant) de la famille des [MASK]. Il est utilisé dans le traitement de l'anxiété lorsque celle-ci s'accompagne de troubles gênants (anxiété généralisée, crise d'angoisse...) et dans le cadre d'un sevrage alcoolique."
translated = translate_fr_en(phrase)
print(translated)

print("Les propositions : ")
for item in fill_mask_original_scibert(translated):
    unmasked = item['sequence'].strip('[SEP]').strip('[CLS]')
    print(unmasked)
    print(translate_en_fr(unmasked))


question_answering_scibert = pipeline(
    "question-answering"
)


print(question_answering_scibert({'question':'What does Bromazepam treat ?','context':'Bromazepam, sold under many brand names, is a benzodiazepine. It is mainly an anti-anxiety agent with similar side effects to diazepam (Valium). In addition to being used to treat anxiety or panic states, bromazepam may be used as a premedicant prior to minor surgery. Bromazepam typically comes in doses of 3 mg and 6 mg tablets. It was patented in 1961 by Roche and approved for medical use in 1974.'}))

nlp = pipeline('question-answering', model='fmikaelian/camembert-base-squad', tokenizer='fmikaelian/camembert-base-squad')

response = nlp({
    'question': "Quel mouvement artistique Claude Monet avait-il créé ?",
    'context': "Claude Monet, né le 14 novembre 1840 à Paris et mort le 5 décembre 1926 à Giverny, est un peintre français et l’un des fondateurs de l'impressionnisme."
})
print(response)
exit(0)