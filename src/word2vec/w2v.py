#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
w2v.py

#using document text data set
python w2v.py -m checkpoint -i txt_exist.paths 

#using tokenized sentences data set
python w2v.py -m models -i preprocessed_sentences.paths 

#testing similary
python src/word2vec/w2v.py -m ../models/ckpt-200.model -c ../config.json -t "he questions about the subjects' self-reported oral health status,"

"""
from __future__ import print_function
import sys
import os
import re
import json
import numpy
import gensim
from gensim.models.fasttext import FastText 
import multiprocessing
from multiprocessing import Pool, Value
from itertools import tee, chain
import time
import logging
from segtok_spans.tokenizer import med_tokenizer

current_path = os.path.dirname(os.path.realpath(__file__))
test_path = os.path.join(os.path.dirname(os.path.dirname(current_path)),"test_data")
sys.path.append(current_path)
from get_files_multi import get_files


def get_check_pt(checkpoint_dir, chunk_size):
    if not os.path.isdir(checkpoint_dir):
        os.system("mkdir -p %s"%checkpoint_dir)
    checkpt_file = os.path.join(checkpoint_dir, "checkpoint")
    if not os.path.isfile(checkpt_file):
        return None, os.path.join(checkpoint_dir, '%s.model'%("ckpt-"+str(chunk_size)))
    else:
        with open(checkpt_file, "r") as fd:
            line = fd.read().split('\n')[0]
        old_name = line.split('"')[1]
        old_ckpt = os.path.join(checkpoint_dir, '%s.model'%old_name)
        ckpt_idx = int(old_name.split('-')[-1]) + chunk_size
        return old_ckpt, os.path.join(checkpoint_dir, '%s.model'%("ckpt-"+str(ckpt_idx)))

counter = None
def split_hyphen_segtok(doc):
    global counter

    (output_dir, f_name, text) = doc    
    def run(text):
        text = text.replace("-", " - ").replace("  ", " ")
        for sentence in split_multi(text.lower()):
            yield "<s>"
            tokens, _ = zip(*med_tokenizer(sentence))
            for token in tokens:
                yield token
            yield "<e>"

    if counter.value % 10 == 0:
        print ("\rpreprocessing files: %3d"%(counter.value), end='    ')
        sys.stdout.flush()

    with counter.get_lock():
        counter.value += 1

    tokens = list(run(text))
    if output_dir != "":
        sentences = " ".join(tokens).replace(' <e> <s> ', '\n').replace('<s> ', '').replace(' <e>', '')
        with open(os.path.join(output_dir, f_name[:-9] + ".tokenized.txt"), "w") as fd:
            fd.write(sentences)
    return tokens


def simply_split(doc):
    global counter

    (output_dir, f_name, text) = doc    
    def run(text):
        for sentence in text.split('\n'):
            yield "<s>"
            for token in sentence.split():
                yield token
            yield "<e>"

    if counter.value % 10 == 0:
        print ("\rpreprocessing files: %3d"%(counter.value), end='    ')
        sys.stdout.flush()

    with counter.get_lock():
        counter.value += 1

    return list(run(text))


def init(args):
    global counter
    counter = args


def read_and_preprocess(input_path, preprocessing_func, output_dir="", \
    input_span=(0, 10000000), verbose=False):

    counter = Value('i', 0)

    with open(input_path, 'r') as fr:
        txt_files = [line.strip() for line in fr if line.strip()]

    (s, e) = input_span
    print ("total docs: %d\nspan      : %d:%d"%(len(txt_files),s, e))
    print('preprocessing ........')
        
    txt_files = txt_files[s:e]

    t0=time.time()
    docs = []
    b = 0
    for f in txt_files[:]:
        with open(f, 'r') as fr:
            doc = fr.read().lower()
            b += len(doc)
            docs += [(output_dir, f.split('/')[-1], doc)]

    if verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    else:
        logging.disable(logging.INFO)
    logging.info("read files     : %d\n%s"%(len(txt_files), str(txt_files[:2])))
    logging.info('total bytes    : %d'%(b))
    logging.info('read files time: %.2f'%(time.time() - t0))

    t0=time.time()    
    document_lst = None
    if docs != []:
        tokenized = docs[0][1].split(".")[-2]
        if tokenized == "tokenized":
            print ("Using tokenized docs!")
            document_lst = []
            for (output_dir, f, doc) in docs:
                tokens = []
                for sent in doc.split('\n'):
                    tokens += ['<s>'] + sent.split() + ['<e>']
                document_lst += [tokens]
        else:
            pool = Pool(initializer=init, initargs=(counter,) )
            document_lst = pool.map_async(preprocessing_func, docs).get()
            pool.close()
            pool.join()        

    logging.info('preprocess time: %.2f'%(time.time() - t0))
    
    return {"documents":document_lst, "n_docs":len(txt_files), \
        "input_files_and_span": (input_path, s, e), "bytes": b}


def read_and_preprocess_dbtxt(input_path, span):
    def run(text):
        for sentence in text.replace('<new_line>', '\n').replace('\t', '\n').replace('. ', '.\n').split('\n'):
            sentence = sentence.strip()
            if re.findall(r"[A-Za-z]+", sentence):
                yield "<s>"
                tokens, _ = zip(*med_tokenizer(sentence))
                for token in tokens:
                    yield token
                yield "<e>"

    document_lst = []
    b = 0
    with open(input_path, 'r') as fr:
        txt_files = [line.strip() for line in fr if line.strip()]
    for path in txt_files:
        with open(path, 'r') as fr:
            text = fr.read()
            b += len(text)
            for report_text in text.split('\n'):
                document_lst += [list(run(report_text))]

    return {"documents":document_lst, "n_docs":len(txt_files), \
        "input_files_and_span": (input_path, span[0], span[1]), "bytes": b}


def train(documents, old_ckpt=None, new_ckpt=None, \
    size=300,
    window=5,
    min_count=1,
    epochs=5,
    workers=multiprocessing.cpu_count(), verbose=False):

    if verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    else:
        logging.disable(logging.INFO)

    #create input data stream
    docs = documents["documents"];print(len(documents["documents"][0]));print(documents["documents"][0],documents.keys());sys.exit(0)
    n_docs = documents["n_docs"]
    docs, docs_vocab = tee(docs)
    docs_train = list(docs)
    
    # build the vocabulary
    t0 = time.time()   
    if old_ckpt == None:
        model = FastText(sg=1, size=size, min_count=min_count, window=window, workers=workers)
        model.build_vocab(docs_vocab)
    else:
        model = gensim.models.Word2Vec.load(old_ckpt)
        model.build_vocab(docs_vocab, update=True)
    logging.info('build vocab time: %.2f'%(time.time() - t0))

    print("n_docs          :", n_docs)
    print("epochs          :", epochs)
    print("size            :", size)
    print("min count       :", min_count)
    print("workers         :", workers)
    print("vocab size      :", len(model.wv.vocab))
    print("new_ckpt        :", new_ckpt)
    print('build vocab time: %.2f'%(time.time() - t0))
    print('training ........')

    t0 = time.time()
    model.train(docs_train, total_examples=n_docs, epochs=epochs)
    logging.info('training time  : %.2f'%(time.time() - t0))
    print('training time   : %.2f'%(time.time() - t0))

    model.save(new_ckpt)

    paths = new_ckpt.split('/')
    checkpoint_dir = '/'.join(paths[:-1])
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint")
    checkpoint_name = '.'.join(paths[-1].split('.')[:-1])
    with open(checkpoint_file_path, 'w') as fw:
        fw.write('model_checkpoint_path: "%s"'%(checkpoint_name))
    print('save_model to   :', new_ckpt)

    if old_ckpt != None:
        paths = old_ckpt.split('/')
        old_checkpoint = paths[-1].split('.')[-2]
        os.system("rm %s/*%s*"%(checkpoint_dir, old_checkpoint))
        print("rm %s/*%s*"%(checkpoint_dir, old_checkpoint))        

    return model

    
from numbers import Number
from collections import Set, Mapping, deque

try: # Python 2
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = 'iteritems'
except NameError: # Python 3
    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = 'items'

def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    def inner(obj, _seen_ids = set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)


def get_similar(text, model_str): #gensim.utils.simple_preprocess):
    tokens = word_tokenizer(text.lower())
    if model_str.endswith(".model"):
        model_path = model_str
    else:
        print("ERROR: invalid model path (w2v.model)", model_str)
    model = gensim.models.Word2Vec.load(model_path)

    ret_str = ''
    for tok in tokens:
        ret_str += "%15s :"%(tok)
        similars = [w for (w, v) in model.wv.most_similar(positive=tok,topn=5)]
        ret_str += ','.join([" %s"%(tok) for tok in similars]) + '\n'
    
    return ret_str


def main():
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--output_dir", dest="output_dir", type=str, default="", help="save tokenized text path (default={})".format(""))
    argparser.add_argument("-i", "--input_path", dest="input_path", type=str, default=os.path.join(test_path, "tokenized_text.paths"),  help="input_path (default={})".format(None))
    argparser.add_argument("-m", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="", help="save checkpoint (default={})".format(None))
    argparser.add_argument("-s", "--span", dest="input_span", type=str, default="", help="span of files list (default={})".format(""))
    argparser.add_argument("--tokenizer", dest="tokenizer", type=str, default="", help="tokenizer segtok/simple (default={})".format(""))
    argparser.add_argument("-c", "--config", dest="config_path", default=os.path.join(test_path, "config.json"), help="config.json (default={})".format("config.json"))
    argparser.add_argument("-v", "--verbose", dest="verbose", default=False, action='store_true', help="verbose")
    argparser.add_argument("-t", "--text", dest="text_str", type=str, default="", help="similar 5 words (default={})".format("abc efg"))
    args = argparser.parse_args()

    with open(args.config_path) as fj:
        config = json.load(fj)

    verbose = args.verbose

    input_path, checkpoint_dir, chunk_size, epochs, size, min_count, tokenizer, verbose = \
        config['input_path'], config['checkpoint_dir'], config['chunk_size'], \
        config['epochs'], config['size'], config['min_count'], config['tokenizer'], config['verbose']

    if args.input_path != "":
        input_path = args.input_path
    if args.checkpoint_dir != "":
        checkpoint_dir = args.checkpoint_dir
    if args.tokenizer != "":
        tokenizer = args.tokenizer
    if args.text_str != "":
        print ("checkpoint_dir", checkpoint_dir, file=sys.stdout)
        print (get_similar(args.text_str, checkpoint_dir), file=sys.stdout)
        sys.exit(0)
    
    input_span = (0, chunk_size)
    if args.input_span != "":
        (s, e) = tuple(args.input_span.split(":"))
        input_span = (int(s), int(e))
    chunk_size = input_span[1] - input_span[0]

    old_ckpt, new_ckpt = get_check_pt(checkpoint_dir, chunk_size)

    if tokenizer == "dbtext":
        docs = read_and_preprocess_dbtxt(input_path, input_span)
    else:
        preprocessing_func = split_hyphen_segtok
        if args.tokenizer == "":
            preprocessing_func = simply_split
        docs = read_and_preprocess(input_path, preprocessing_func, args.output_dir, input_span=input_span, verbose=verbose)

    if docs["n_docs"] == 0:
        print ("ERROR: Empty dataset! input_span:", str(input_span))
        sys.exit(0)


    #print (type(docs["documents"]), docs["n_docs"], docs["input_files_and_span"], docs["bytes"])
    print (old_ckpt, new_ckpt, input_span, chunk_size, verbose)

    model = train(docs, size=size, min_count=min_count, \
        old_ckpt=old_ckpt, \
        new_ckpt=new_ckpt, \
        epochs=epochs, verbose=verbose)

    print (model)
    print ("model size:", getsize(model))

if __name__ == '__main__':
    main()

