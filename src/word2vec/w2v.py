#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
w2v.py

#using document text data set 
python w2v.py -m checkpoint -i txt.paths -m model_path

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
from segtok_spans.segmenter import split_multi
from segtok_spans.tokenizer import med_tokenizer

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_path))
#from word2vec import getsize


def get_check_pt(checkpoint_dir, chunk_size):
    if checkpoint_dir.endswith(".model"):
        checkpoint_dir = "/".join(checkpoint_dir.split("/")[:1])
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
def to_tokens(doc):
    global counter

    f_name, text, tokenized_save_dir = doc    
    def run_tokenizer(text):
        #text = text.replace("-", " - ").replace("  ", " ")
        for sentence, _ in split_multi(text.lower()):
            if re.findall(r"[A-Za-z]+", sentence):
                yield "<s>"
                sentence = sentence.replace(".", " . ").replace("  ", " ")
                tokens, _ = zip(*med_tokenizer(sentence))
                for token in tokens:
                    yield token
                yield "<e>"

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

    if f_name.endswith(".tokenized.txt"):
        tokens = list(run(text))
    else:
        tokens = list(run_tokenizer(text))
        if tokenized_save_dir != "":
            sentences = " ".join(tokens).replace(' <e> <s> ', '\n').replace('<s> ', '').replace(' <e>', '')
            with open(os.path.join(tokenized_save_dir, ".".join(f_name.split(".")[:-1]) + ".tokenized.txt"), "w") as fd:
                fd.write(sentences)

    return tokens


def init(args):
    global counter
    counter = args


def read_and_preprocess(input_path, tokenized_save_dir="", input_span=(0, 10000000), verbose=False):
    global counter
    counter = Value('i', 0)
    
    if input_path.endswith(".paths"):
        with open(input_path, 'r') as fr:
            txt_files = [line.strip() for line in fr if line.strip()]
    elif os.path.isdir(input_path):
        txt_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".txt")]

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
            #clean from dbtext
            if doc.strip():
                doc = doc.replace('<new_line>', '\n').replace('\t', '\n').replace('. ', '.\n').strip()
                docs += [(f.split('/')[-1], doc, tokenized_save_dir)]
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
        pool = Pool(initializer=init, initargs=(counter,) )
        document_lst = pool.map_async(to_tokens, docs).get()
        #document_lst = [to_tokens(doc) for doc in docs]
        pool.close()
        pool.join()        
    if tokenized_save_dir:
        f_names = [os.path.join(tokenized_save_dir, ".".join(x[0].split(".")[:-1]) + ".tokenized.txt")  for x in docs if not x[0].endswith(".tokenized.txt")]
        with open(os.path.join(tokenized_save_dir, "tokenized.paths"), "w") as fd:
            fd.write("\n".join(f_names))
        print("save to :", os.path.join(tokenized_save_dir, "tokenized.paths"))

    logging.info('preprocess time: %.2f'%(time.time() - t0))
    
    return {"documents":document_lst, "n_docs":len(txt_files), "input_files_and_span": (input_path, s, e), "bytes": b}


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
    docs = documents["documents"]
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


def main():
    import argparse
    sys.path.append(current_path)
    from helper import getsize

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--output_tokenized_dir", dest="output_tokenized_dir", type=str, default="", help="save tokenized text path")
    argparser.add_argument("-i", "--input_path", dest="input_path", type=str, default="",  help="input paths file")
    argparser.add_argument("-m", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="", help="save checkpoints")
    argparser.add_argument("-l", "--limit", dest="limit", type=str, default="", help="span of files, eg. 10:1000")
    argparser.add_argument("-c", "--config", dest="config_path", default=os.path.join(current_path, "resources/config.json"), help="config.json (default={})".format("config.json"))
    argparser.add_argument("-v", "--verbose", dest="verbose", default=False, action='store_true', help="verbose")
    args = argparser.parse_args()

    with open(args.config_path) as fj:
        config = json.load(fj)

    verbose = args.verbose

    input_path, checkpoint_dir, chunk_size, epochs, size, min_count, output_tokenized_dir, verbose = \
        config['input_path'], config['checkpoint_dir'], config['chunk_size'], \
        config['epochs'], config['size'], config['min_count'], config['output_tokenized_dir'], config['verbose']

    if args.input_path != "":
        input_path = args.input_path
    if args.checkpoint_dir != "":
        checkpoint_dir = args.checkpoint_dir
    if args.output_tokenized_dir:
        output_tokenized_dir = args.output_tokenized_dir
    if output_tokenized_dir:
        if not os.path.isdir(output_tokenized_dir):
            os.system("mkdir %s"%output_tokenized_dir)
    
    limit = (0, chunk_size)
    if args.limit:
        lst = args.limit.split(":")
        if len(lst) == 2:
            s, e = tuple(lst)
        else:
            s, e = 0, lst[0]
        limit = (int(s), int(e))
    chunk_size = limit[1] - limit[0]
    old_ckpt, new_ckpt = get_check_pt(checkpoint_dir, chunk_size)

    docs = read_and_preprocess(input_path, output_tokenized_dir, input_span=limit, verbose=verbose)
    if docs["n_docs"] == 0:
        print ("ERROR: Empty dataset! limit:", str(limit))
        sys.exit(0)
    #print (type(docs["documents"]), docs["n_docs"], docs["input_files_and_span"], docs["bytes"])
    print (old_ckpt, new_ckpt, limit, chunk_size, verbose)

    model = train(docs, size=size, min_count=min_count, \
        old_ckpt=old_ckpt, \
        new_ckpt=new_ckpt, \
        epochs=epochs, verbose=verbose)

    print (model)
    print ("model size:", getsize(model))


if __name__ == '__main__':
    main()

