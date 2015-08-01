#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
A module for globally sharing Word Vectors

Miao Li
limiaogg@126.com
'''
import sys, time
from gensim.models import Word2Vec

_WORDVEC_MODEL = None

def InitModel():
    global _WORDVEC_MODEL
    assert (_WORDVEC_MODEL == None),'InitModel has already been called.'
    print >>sys.stderr, "Loading Word2Vec Models ..."
    start = time.time()
    _WORDVEC_MODEL = Word2Vec.load_word2vec_format('/home/limiao/open_tools/Word2Vec/models/wiki_en_models/wiki.en.text.vector', binary=False)
    end = time.time()
    print >>sys.stderr, "Completed! time: ", end-start, "sec."

def GetModel():
    global _WORDVEC_MODEL
    if _WORDVEC_MODEL == None:
        InitModel()
    return _WORDVEC_MODEL

   