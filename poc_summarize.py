from transformers import *
import sys
# Load model, model config and tokenizer via Transformers
custom_config = AutoConfig.from_pretrained('fmikaelian/camembert-base-squad')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('fmikaelian/camembert-base-squad')
custom_model = AutoModel.from_pretrained('fmikaelian/camembert-base-squad', config=custom_config)

from summarizer import Summarizer

body = "L'Espagne est bordée au nord-est par les Pyrénées, qui constituent une frontière naturelle avec la France et l'Andorre ; à l'est et au sud-est par la mer Méditerranée, au sud-sud-ouest par le territoire britannique de Gibraltar et le détroit du même nom, ce dernier séparant le continent européen de l'Afrique. Le Portugal est limitrophe de l'Espagne à l'ouest tandis que l'océan Atlantique borde le pays à l'ouest-nord-ouest ; enfin le golfe de Gascogne baigne le littoral nord. Le territoire espagnol inclut également les îles Baléares en Méditerranée, les îles Canaries dans l'océan Atlantique au large de la côte africaine, et deux cités autonomes en Afrique du Nord, Ceuta et Melilla, limitrophes du Maroc. Avec une superficie de 504 030 km2, l'Espagne est le pays le plus étendu d'Europe de l'Ouest et de l'Union européenne après la France ainsi que le troisième d'Europe derrière l'Ukraine et la France si l'on exclut la partie européenne (selon les définitions) de la Russie."
body2 = 'Something else you want to summarize with BERT'
model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
print(model(body))
model(body2)

sys.exit(0)