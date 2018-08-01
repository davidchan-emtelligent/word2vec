from __future__ import print_function
import sys
import os
import multiprocessing

default_in_dir = "../../cas-reader/output_dir -o tokenized_sentences"
default_in_dir = "/shared/dropbox/ctakes_conll/output/clinical/dc_summaries"
default_out_dir = "tokenized_sentences"

def init(args):
	global counter
	counter = args

def func(x):
	global counter

	(i, in_dir, out_dir, f) = x
	with open(os.path.join(in_dir,f), 'r') as fd:
		lines = fd.read().split('\n')

	tokens = [line.split(' ')[0] for line in lines]
	doc_str = " ".join(tokens).replace("  ", "\n").strip()

	if doc_str == "":
		return None
	
	out_dir = os.path.join(out_dir,f)
	with open(out_dir, 'w') as fd:
		fd.write(doc_str)

	if counter.value % 10 == 0:
		print ("\rfiles:%2d"%(i), end="	")
		sys.stdout.flush()

	with counter.get_lock():
		counter.value += 1

	return  out_dir


if __name__ == '__main__':
	import argparse

	argparser = argparse.ArgumentParser()

	argparser.add_argument("-i", "--input_dir", dest="input_dir", type=str, default=default_in_dir, \
		help="input_dir (default={})".format(None))

	argparser.add_argument("-o", "--output_dir", dest="output_dir", type=str, default=default_out_dir, \
		help="output_dir (default={})".format(None))

	argparser.add_argument("-l", "--limit", dest="limit", type=int, default=2000000, \
		help="limit (default={})".format(2000000))

	args = argparser.parse_args()

	in_dir = args.input_dir
	out_dir = args.output_dir

	idx_fs = [ (i, in_dir, out_dir, f) for i, f in enumerate(os.listdir(in_dir)[:args.limit])]
	#ret_lst = [func(x) for x in idx_fs]
	counter = multiprocessing.Value('i', 0)
	ret_lst = multiprocessing.Pool(initializer=init, initargs=(counter,) ).imap_unordered(func, idx_fs)
	ret_lst = [r for r in ret_lst if r != None]
	print ("\ntotal files:", len(ret_lst), "\nsave to:", out_dir)
