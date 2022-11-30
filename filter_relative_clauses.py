import spacy
import json
from nltk import Tree
from spacy.pipeline import DependencyParser
#from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL

# spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")

# config = {
#    "moves": None,
#    "update_with_oracle_cut_size": 100,
#    "learn_tokens": False,
#    "min_action_freq": 30,
#    "model": DEFAULT_PARSER_MODEL,
# }
# nlp.add_pipe("parser", config=config)
# parser = nlp.add_pipe("parser", config=config) # DependencyParser(nlp.vocab, model)

# parser = nlp.add_pipe("parser")

sentences = [
"This customer who had visited most children has worn some shoes.",
"This boy that didn't question all guests has forgotten about what wasn't vaporizing.",
"Boys that aren't disturbing Natalie suffer.",
"Some customers who irritates Amelia do persuade Galileo to come here.",
"Many children that had questioned this government do cooperate.",
"All pedestrians that wouldn't shock William read."]


def to_nltk_tree(node):
   if node.n_lefts + node.n_rights > 0:
      return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
   else:
      return node.orth_


for sentence in sentences:
   doc = nlp(sentence)

   print(sentence)

   for token in doc:
      print(token.text, token.dep_, token.head.text, token.head.pos_)

   # spacy.displacy.serve(doc, style="dep")

   [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
   print('=============')


# with open('data/distractor_agreement_relative_clause.jsonl', 'r') as inf:
#     data = json.load(inf)
#     for i in range(0, 6):
#         print(data)
# json.decoder.JSONDecodeError: Extra data:


# https://demos.explosion.ai/displacy