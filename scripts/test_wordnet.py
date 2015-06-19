from nltk.corpus import wordnet as wn
import sys

while True:
  word = raw_input('input a word: ')
  if word == '#':
    break
  for synset in wn.synsets(word):
    print synset
    for item in synset.lemma_names:
      print item + ',' ,
    print

