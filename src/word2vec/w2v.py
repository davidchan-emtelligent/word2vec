#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
w2v.py

#using document text data set
python w2v.py -m models -i txt_exist.paths --save_dir preprocessed_sentences

#using tokenized sentences data set
python w2v.py -m models -i preprocessed_sentences.paths 

#testing similary
python w2v.py -m models -t "he questions about the subjects' self-reported oral health status,"

"""
from __future__ import print_function
import sys
import os
import json
import gensim
from gensim.models.fasttext import FastText 
import multiprocessing
from multiprocessing import Pool, Value
from itertools import tee, chain
import time
import logging

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
from get_files_multi import get_files

def get_latest(model_root, size):
	end_size_lastest = []
	for x in os.listdir(model_root):
		(e, d) = tuple(x.split('_')[0].split('d') )
		end_size_lastest += [(int(e), int(d), x)]

	if end_size_lastest == []:
		return (0, size, None)

	return sorted(end_size_lastest)[-1]


from segtok.segmenter import split_multi
from segtok.tokenizer import word_tokenizer, split_contractions

counter = None
def split_hyphen_segtok(doc):
	global counter

	(save_dir, f_name, text) = doc	
	def run(text):
		text = text.lower().replace("-", " - ").replace("  ", " ")
		for sentence in split_multi(text.lower()):
			yield "<s>"
			for token in split_contractions(word_tokenizer(sentence)):
				yield token
			yield "<e>"

	if counter.value % 10 == 0:
		print ("\rpreprocessing files: %3d"%(counter.value), end='    ')
		sys.stdout.flush()

	with counter.get_lock():
		counter.value += 1

	tokens = list(run(text))
	if save_dir != "":
		sentences = " ".join(tokens).replace(' <e> <s> ', '\n').replace('<s> ', '').replace(' <e>', '')
		with open(os.path.join(save_dir, f_name[:-9] + ".tokenized.txt"), "w") as fd:
			fd.write(sentences)
	return tokens


def simply_split(doc):
	global counter

	(save_dir, f_name, text) = doc	
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


def read_and_preprocess(input_path, preprocessing_func, save_dir="", \
	input_span=(0, 10000000), verbose=False):

	counter = Value('i', 0)

	with open(input_path, 'r') as fr:
		txt_files = fr.read().split('\n')

	(s, e) = input_span
	print ("total docs: %d\nspan	  : %d:%d"%(len(txt_files),s, e))
	print('preprocessing ........')
		
	txt_files = txt_files[s:e]

	t0=time.time()
	docs = []
	b = 0
	for f in txt_files[:]:
		with open(f, 'r') as fr:
			doc = fr.read()
			b += len(doc)
			docs += [(save_dir, f.split('/')[-1], doc)]

	if verbose:
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	else:
		logging.disable(logging.INFO)
	logging.info("read files	 : %d\n%s"%(len(txt_files), str(txt_files[:2])))
	logging.info('total bytes	: %d'%(b))
	logging.info('read files time: %.2f'%(time.time() - t0))

	t0=time.time()	
	document_lst = None
	if docs != []:
		tokenized = docs[0][1].split(".")[-2]
		if tokenized == "tokenized":
			print ("Using tokenized docs!")
			document_lst = []
			for (save_dir, f, doc) in docs:
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


def train(documents, save_model=None, retrain=None, \
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
	if retrain == None:
		model = FastText(sg=1, size=size, min_count=min_count, window=window, workers=workers)
		model.build_vocab(docs_vocab)
	else:
		model = gensim.models.Word2Vec.load(retrain)
		model.build_vocab(docs_vocab, update=True)
	logging.info('build vocab time: %.2f'%(time.time() - t0))

	print("n_docs          :",n_docs)
	print("epochs          :",epochs)
	print("size            :",size)
	print("workers         :",workers)
	print("vocab size      :",len(model.wv.vocab))
	print("retrain         :",retrain)
	print('build vocab time: %.2f'%(time.time() - t0))
	print('training ........')

	t0 = time.time()
	model.train(docs_train, total_examples=n_docs, epochs=epochs)
	logging.info('training time  : %.2f'%(time.time() - t0))
	print('training time   : %.2f'%(time.time() - t0))

	if save_model != None:
		model.save(save_model)
		paths = save_model.split('/')
		txt_path = '/'.join(paths[:-1] + [paths[-1].split('.')[0] + '.doc_span'])
		(input_path, s,e) = documents["input_files_and_span"]
		with open(txt_path, 'w') as fw:
			fw.write("%s %d:%d"%(input_path, s, e))
		print('save_model to   :', save_model)

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


def get_similar(text, model_str, size): #gensim.utils.simple_preprocess):
	tokens = preprocessing_func(text)
	if model_str.endswith(".model"):
		model_path = model_str
	else:
		(e, size, retrain) = get_latest(model_root, size)
		model_path =os.path.join(model_root, retrain, '%s.model'%('d' + 'd'.join(retrain.split('d')[1:])))
	model = gensim.models.Word2Vec.load(model_path)

	ret_str = ''
	for tok in tokens:
		ret_str += "%15s :"%(tok)
		similars = [w for (w, v) in model.wv.most_similar(positive=tok,topn=5)]
		ret_str += ','.join([" %s"%(tok) for tok in similars]) + '\n'
	
	return ret_str


if __name__ == '__main__':
	import argparse

	argparser = argparse.ArgumentParser()

	argparser.add_argument("-s", "--save_dir", dest="save_dir", type=str, default="", \
		help="save tokenized text path (default={})".format(""))

	argparser.add_argument("-i", "--input_path", dest="input_path", type=str, default=None, \
		help="input_path (default={})".format(None))

	argparser.add_argument("-m", "--model_root", dest="model_root", type=str, default=None, \
		help="save model root (default={})".format(None))

	argparser.add_argument("--tokenizer", dest="tokenizer", type=str, default="", \
		help="tokenizer (default={})".format("segtok"))

	argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=None, \
		help="epochs (default={})".format(None))

	argparser.add_argument("-c", "--config", dest="config_path", default="config.json", \
		help="config.json (default={})".format("config.json"))

	argparser.add_argument("-v", "--verbose", dest="verbose", default=False, action='store_true',\
		help="verbose")

	argparser.add_argument("-r", "--resume", dest="resume", default=False, action='store_true',\
		help="retrain")

	argparser.add_argument("-t", "--text", dest="text_str", type=str, default="", \
		help="similar 5 words (default={})".format("abc efg"))

	args = argparser.parse_args()

	with open(args.config_path) as fj:
		config = json.load(fj)

	verbose = args.verbose; print (config)

	input_path, model_root, chunk_size, epochs, resume, size, verbose = \
		config['input_path'], config['model_root'], config['chunk_size'], \
		config['epochs'], config['resume'], config['size'], config['verbose']

	if args.input_path != None:
		input_path = args.input_path
	if args.model_root != None:
		model_root = args.model_root
	if args.epochs != None:
		epochs = args.epochs
	if args.resume != False:
		resume = args.resum
	if args.text_str != "":
		print ("model_root", model_root, file=sys.stdout)
		print (get_similar(args.text_str, model_root, size), file=sys.stdout)
		sys.exit(0)

	input_span = (0, chunk_size)
	retrain = None
	if resume:
		(e, size, retrain) = get_latest(model_root, size)
		input_span = (e, e+chunk_size)
	if retrain != None:
		retrain=os.path.join(model_root, retrain, '%s.model'%('d'+'d'.join(retrain.split('d')[1:])))

	preprocessing_func = split_hyphen_segtok;print (args.tokenizer)
	if args.tokenizer == "simple":
		preprocessing_func = simply_split

	docs = read_and_preprocess(input_path, preprocessing_func, args.save_dir, input_span=input_span, verbose=verbose)
	if docs["n_docs"] == 0:
		print ("ERROR: Empty dataset! input_span:", str(input_span))
		sys.exit(0)
		
	model_name = "d%d_model"%(size)
	save_model_dir = os.path.join(model_root, str(input_span[1])+model_name)
	if not os.path.isdir(save_model_dir):
		os.system("mkdir %s"%save_model_dir)

	#print (type(docs["documents"]), docs["n_docs"], docs["input_files_and_span"], docs["bytes"])
	#print (save_model_dir, model_name)

	model = train(docs, size=size, \
		save_model=os.path.join(save_model_dir, '%s.model'%(model_name)), \
		retrain=retrain, \
		epochs=epochs, verbose=verbose)

	print (model)
	print (getsize(model))  
