#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
	python helper.py -i <input_model> -o <save_file>           	#create save_file to store vectors

	python helper.py -i <input_dir> -o <output_dir> -j cp_3_10000	#cp input_dir to sentences/sentences_3, 4, ..
	python helper.py -i pmc_preprocessed/sentences_1 -o /shared2/data/PMC/sentences -j cp_0

	python helper.py -t "he questions about the subjects' self-reported oral health status,"
	python helper.py -t "self-reorted"
"""
from __future__ import print_function
import sys
import os
import json
import string
import gensim
import multiprocessing

default_input_path = "/shared/data/PMC/w2v_models/1200000d300_model/d300_model.model"
#default_input_path = "models/100000d200_model/d200_model.model"
default_text = "he questions about the subjects' self-reported oral health status,"

def vec_work(w):
	vec = model.wv[w]

	try:
		w = w.encode('utf-8')
	except:
		pass

	ret = ''
	for i in range(vec.shape[0]):
		ret += " %.5f"%(vec[i])
	return "%s%s"%(w, ret)


def save_vec(model, output_path):
	ws = list(model.wv.vocab)
	#lines = [vec_work(w) for w in ws]
	lines = list(multiprocessing.Pool().imap_unordered(vec_work, ws))
	
	with open(output_path, 'wb') as fd:
		fd.write('\n'.join(lines))

	return "vec saved to:" + output_path


def get_similar(model, text):
	for p in string.punctuation:
		text = text.replace(p, " "+p+" ")
	text = text.replace("  ", " ").replace("  ", " ")
	tokens = text.split()

	ret_str = ''
	for tok in tokens:
		ret_str += "%15s :"%(tok)
		similars = [w for (w, v) in model.wv.most_similar(positive=tok,topn=5)]
		ret_str += ','.join([" %s"%(tok) for tok in similars]) + '\n'
	
	return ret_str


def get_vec(model, token, verbose=False):
	if verbose:
		if token not in list(model.wv.vocab):
			print ('"%s": not in w2v model.'%(token))

	return model.wv[token]

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
	counter = multiprocessing.Value('i', 0)
	fs = os.listdir(one_level_dir)[:limit]
	spans = [ (s, s+step) for s in range(0, len(fs), step)]
	n_batch = len(spans)
	print ("files:", len(fs), " sub_folers:", len(spans), " output_path:", split_dir)
	dir_name = split_dir.split('/')[-1]
	out_dir = [os.path.join(split_dir, dir_name+"_%d"%(i+idx_start)) for i in range(n_batch)]
	for dir_name in out_dir:
		ret = os.system("mkdir -p %s"%(dir_name)	)

	fpath_fs = [(one_level_dir, out_dir[i], op, fs[s:e]) for i, (s,e) in enumerate(spans)]
	
	pool = multiprocessing.Pool(initializer=init, initargs=(counter,) )
	ret_lst = pool.map(work, fpath_fs)
	
	return ret_lst


if __name__ == '__main__':
	import argparse

	argparser = argparse.ArgumentParser()

	argparser.add_argument("-i", "--input_path", dest="input_path", type=str, default=default_input_path, \
		help="input_path (default={})".format(default_input_path))

	argparser.add_argument("-o", "--output_path", dest="output_path", type=str, default=None, \
		help="output_path (default={})".format(None))

	argparser.add_argument("-j", "--job", dest="job", type=str, default="", \
		help='job (default="{}")'.format(""))

	argparser.add_argument("-t", "--text", dest="text", type=str, default=default_text, \
		help='similar 5 words (default="{}")'.format(default_text))

	argparser.add_argument("-v", "--verbose", dest="verbose", default=True, action='store_true',\
		help="verbose")

	args = argparser.parse_args()

	input_path = args.input_path
	
	if os.path.isfile(input_path) or input_path[-6:] == ".model":
		print ("loading model:", input_path, " .....")
		model = gensim.models.Word2Vec.load(input_path)

	if args.output_path != None:
		output_path = args.output_path
		job = args.job
		if job == "":
			print (save_vec(model, output_path))
		else:
			op = 'cp'
			idx_start = 0
			lst = job.split('_')
			if len(lst) > 1:
				(op, idx_start,step) = tuple(lst)
				idx_start, step = int(idx_start), int(step)
			print ("\n".join(split_dir(input_path, output_path, op=op, idx_start=idx_start, step=step)))
	else:
		tokens = args.text.split()
		if len(tokens) == 1:
			vec = " ".join(["%.6f"%(v) for v in get_vec(model, tokens[0], args.verbose)[:5]])
			print (vec+" .....")
		elif len(tokens) > 0:
			print (get_similar(model, args.text))



