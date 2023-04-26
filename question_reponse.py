from transformers import pipeline
import sys


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


translated_question = translate_fr_en("Quelles sont les causes des calculs urinaires ?")
print("Question")
print(translated_question)
print("Context")
translated_context = translate_fr_en("Une vessie qui ne se vide pas correctement augmente le risque d'infection urinaire et de problèmes rénaux. Certains hommes qui souffrent d'hypertrophie bénigne de la prostate sont victimes de calculs urinaires dans la vessie ou d'infections à répétition. On observe parfois des problèmes de rétention urinaire et, de manière très occasionnelle, une obturation complète de l'urètre qui constitue alors une urgence médicale : c'est la rétention d'urines aiguë. Chez certains patients, les problèmes urinaires liés à l'HBP ont des conséquences psychologiques négatives sur leur sexualité. L’hypertrophie bénigne de la prostate ne dégénère jamais en cancer de la prostate. Néanmoins, les deux maladies peuvent présenter de symptômes similaires bien que, la plupart du temps, le cancer de la prostate ne provoque aucun symptôme. Il est possible de souffrir à la fois d’un adénome de la prostate et d’un cancer de la prostate.")
print(translated_context)
nlp = pipeline('question-answering', model='allenai/scibert_scivocab_uncased', tokenizer='allenai/scibert_scivocab_uncased')
#allenai/scibert_scivocab_uncased
response = nlp({
    'question': translated_question,
    'context': translated_context
})
print(response)
print(response['answer'])

sys.exit()