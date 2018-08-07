#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1) extract word count from tokenized text
	python src/word2vec/helper.py -i <input_dir> -o <output_path> -j <job>
	python src/word2vec/helper.py -i sentences -o word_count.txt -j wc

2) split files of a dir into sub_dirs
	python src/word2vec/helper.py -i <input_dir> -o <output_dir> -j <op_start_steps>
	python src/word2vec/helper.py -i tokenized_text -o ds_sentences -j cp_0_10000

3) extract w2v from model
	python src/word2vec/helper.py -m <input_model> -i <input_vocab_path> -o <save_w2v_file> -j <job>
	python src/word2vec/helper.py -m <input_model> -o <save_w2v_file> -j w2v

4) test model: get vectors or most similar:
	python src/word2vec/helper.py -m <input_model> -t "he questions about the subjects' self-reported oral health status,"
	python src/word2vec/helper.py -m <input_model> -t "self-reorted" -j vec

"""
from __future__ import print_function
import sys
import os
import json
import string
import gensim
import multiprocessing
from itertools import groupby
from random import shuffle

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
from get_files_multi import get_files

default_model_path = "/shared/data/PMC/w2v_models/1300000d300_model/d300_model.model"
default_text = "he questions about the subjects' self-reported oral health status,"

#1)save word count, wc
def merge_token_count(token_count_lst):
	token_count = []
	for t_cs in token_count_lst:
		token_count += t_cs

	token_count = sorted(token_count, key=lambda x: x[0])
	merged = []
	for k, g in groupby(token_count, lambda x: x[0]):
		count = sum([ x[1] for x in g])
		merged += [(k, count)]

	return merged


def count_work(fs_lst):

	token_count_lst = []
	for f in fs_lst:
		with open(f, 'r') as fd:
			tokens = fd.read().lower().split()
		token_count = []
		for k, g in groupby(sorted(tokens)):
		    token_count += [(k, len(list(g)))]

		token_count_lst += [token_count]

	return merge_token_count(token_count_lst)


def save_word_count(input_tokenized_text_paths, output_path):

	with open(input_tokenized_text_paths, "r") as fd:
		lines = fd.read().split('\n')
	fs = [line for line in lines if lines != ""]

	n_files = len(fs)
	batch_size = int(n_files/99)
	batch_span = [(s, s+batch_size) for s in list(range(0, n_files, batch_size))]
	if batch_span[-1][1] < n_files:
		batch_span += [(batch_span[-1][1], n_files)]

	batches = [fs[s:e] for (s, e) in batch_span]
	print ("total files:", n_files)
	print ("n_batches  :", len(batches), " last batch :",batch_span[-1])

	#token_count = [count_work(batch) for batch in batches]
	token_count = list(multiprocessing.Pool().imap_unordered(count_work, batches))
	merged = ["%s %d"%(w, c) for (w, c) in sorted(merge_token_count(token_count), key=lambda x: x[1], reverse=True)]

	with open(output_path, 'w') as fd:
		fd.write('\n'.join(merged))

	return "word_count saved to: " + output_path


#2) split files of a dir into sub_dirs: cp_2_1000
def init(args):
	global counter
	counter = args

def work(data):
	global counter

	(one_level_dir, out_dir, op, fnames) = data
	for f in fnames:
		#print ("%s %s/%s %s/%s"%(op, one_level_dir, f, out_dir, f))
		os.system("%s %s/%s %s/%s"%(op, one_level_dir, f, out_dir, f))

		if counter.value % 10 == 0:
			print ("\rfiles: %3d"%(counter.value), end='    ')
			sys.stdout.flush()

		with counter.get_lock():
			counter.value += 1

	return out_dir


#def split_dir(one_level_dir, split_dir, op='cp', idx_start=0, step=100, limit=650):
def split_dir(one_level_dir, split_dir, op='cp', idx_start=0, step=10000, limit=2000000):

	fs = os.listdir(one_level_dir)[:limit]
	shuffle(fs)
	spans = [ (s, s+step) for s in range(0, len(fs), step)]
	n_batch = len(spans)
	print ("files:", len(fs), " sub_folders:", len(spans), " output_path:", split_dir)
	dir_name = split_dir.split('/')[-1]
	out_dir = [os.path.join(split_dir, dir_name+"_%d"%(i+idx_start)) for i in range(n_batch)]
	for dir_name in out_dir:
		ret = os.system("mkdir -p %s"%(dir_name)	)

	fpath_fs = [(one_level_dir, out_dir[i], op, fs[s:e]) for i, (s,e) in enumerate(spans)]

	counter = multiprocessing.Value('i', 0)
	pool = multiprocessing.Pool(initializer=init, initargs=(counter,) )
	ret_lst = pool.map(work, fpath_fs)

	print ("")
	return ret_lst


#3) extract w2v from model: w2v
def vec_work(b_m):
	global counter
	global n_vocab
	(batch, model) = b_m

	def get_row(w):
		try:
			if w == "<start>":
				vec = model.wv["<s>"]
			elif w == "<stop>":
				vec = model.wv["<e>"]
			else:
				vec = model.wv[w.lower()]
		except Exception as e:
			print ("WARNING:",[w], str(e))
			return None

		try:
			w = w.decode('utf-8')
		except:
			pass

		ret = ''
		for i in range(vec.shape[0]):
			ret += " %.5f"%(vec[i])

		if counter.value % 100 == 0:
			print ("\rwords: %3d/%d"%(counter.value, n_vocab.value), end='    ')
			sys.stdout.flush()

		with counter.get_lock():
			counter.value += 1

		return "%s%s"%(w, ret)
	
	ret_lst = []
	for w in batch:
		if w == "":
			continue
		w = w.split()[0]  #in case is wc

		row = get_row(w)
		if row != None:
			ret_lst += [row]

	return ret_lst


def save_w2v(model, ws, output_path):

	ws = [w.split()[0] for w in ws]
	if ws[0][0] != "<":
		ws = ["<start>", "<stop>", "<unk>", "<UNK>"] + ws

	output_vocab = '.'.join(output_path.split('.')[:-1]) + "_vocab.txt"
	with open(output_vocab, 'w') as fd:
		fd.write('\n'.join(ws))
	
	len_1 = len(ws)
	batch_size = len_1/7
	spans = [(s, s+batch_size) for s in range(0, len_1, batch_size)]
	data_lst = [(ws[s:e], model) for (s, e) in spans]

	counter = multiprocessing.Value('i', 0)
	n_vocab = multiprocessing.Value('i', len_1)
	pool = multiprocessing.Pool(initializer=init, initargs=(counter,) )

	#vecs = [vec_work(w) for w in ws]
	vecs_lst = pool.map(vec_work, data_lst)
	vecs = []
	for vec in vecs_lst:
		vecs += vec

	with open(output_path, 'w') as fd:
		fd.write('\n'.join(vecs))

	return "\nvocabs saved to: %d\nw2v saved to   :%d"%(output_vocab, output_path)


#4) test model
def get_text_similar(model, text):
	for p in string.punctuation:
		text = text.replace(p, " "+p+" ")
	text = text.replace("  ", " ").replace("  ", " ")
	tokens = text.split()

	ret_str = ''
	for tok in tokens:
		ret_str += "%15s :"%(tok)
		try:
			similars = [w for (w, v) in model.wv.most_similar(positive=tok,topn=5)]
			ret_str += ','.join([" %s"%(tok) for tok in similars]) + '\n'
		except Exception as e:
			ret_str += " ERROR: %s"%e + '\n'

	return ret_str


#5) get vec for a token
def get_vec(model, token, verbose=False):
	if verbose:
		if token not in list(model.wv.vocab):
			print ('"%s": not in w2v model.'%(token))

	return model.wv[token]


if __name__ == '__main__':
	import argparse

	argparser = argparse.ArgumentParser()

	argparser.add_argument("-i", "--input_path", dest="input_path", type=str, default=None, \
		help="input_path (default={})".format(None))

	argparser.add_argument("-o", "--output_path", dest="output_path", type=str, default=None, \
		help="output_path (default={})".format(None))

	argparser.add_argument("-l", "--limit", dest="limit", type=int, default=2000000, \
		help="limit (default={})".format(2000000))

	argparser.add_argument("-m", "--model_path", dest="model_path", type=str, default=default_model_path, \
		help="model_path (default={})".format(default_model_path))

	argparser.add_argument("-j", "--job", dest="job", type=str, default="", \
		help='job: cp_3_100/wc/w2v/vec/smiliar (default="{}")'.format(""))

	argparser.add_argument("-t", "--text", dest="text", type=str, default=default_text, \
		help='similar 5 words (default="{}")'.format(default_text))

	argparser.add_argument("-v", "--verbose", dest="verbose", default=True, action='store_true',\
		help="verbose")

	args = argparser.parse_args()

	input_path = args.input_path
	output_path = args.output_path
	model_path = args.model_path
	job = args.job
	if job == "":
		print("ERRROR: no job")

	#1) extract vocab from tokenized text
	if job == "wc":
		if not os.path.isfile(input_path):
			print("ERRROR: no input_tokenized_text.paths for tokenized files")
			sys.exit(0)
		if output_path ==None:
			print("ERRROR: no output_wc_file")
			sys.exit(0)
		print (save_word_count(input_path, output_path))
		sys.exit(0)

	#2) split files of a dir to more sub_dir. eg. sentences -> sentences/sentences_1, sentences_2, ...
	elif len(job.split("_")) > 1:
		if not os.path.isdir(input_path):
			print("ERRROR: no input_dir files")
			sys.exit(0)
		if output_path ==None:
			print("ERRROR: no output_dir")
		if not os.path.isdir(output_path):
			os.system("mkdir %s"%output_path)
		op = 'cp'
		idx_start = 0
		lst = job.split('_')
		(op, idx_start,step) = tuple(lst)
		idx_start, step = int(idx_start), int(step)

		print ("\n".join(split_dir(input_path, output_path, op=op, idx_start=idx_start, step=step, limit=args.limit)))
		sys.exit(0)

	if model_path != None:
		print ("loading model:", model_path, " .....")
		model = gensim.models.Word2Vec.load(model_path)
		vocab_lst = list(model.wv.vocab)
	else:
		print("ERRROR: no model file")
		sys.exit(0)

	#3) extract w2v from model
	if job == "w2v":
		if output_path == None:
			output_path = model_path.split("/") + "_w2v.txt"
		if input_path != None:
			if not os.path.isfile(input_path):
				print ("ERROR: not exists!", input_path)
				sys.exit(0)
			with open(input_path, 'r') as fd:
				vocab_lst = fd.read().strip().split('\n')

		print (save_w2v(model, vocab_lst, output_path))

	#4) test model with similarity words
	elif job == "vec":
		tokens = args.text.split()
		for tok in tokens:
			vec = " ".join(["%.6f"%(v) for v in get_vec(model, tok, args.verbose)[:5]])
			print (vec+" .....")
	else:
		print (get_text_similar(model, args.text))




