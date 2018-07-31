from __future__ import print_function
import sys
import os
import multiprocessing as multiprocess

default_in_dir = "../../cas-reader/output_dir -o tokenized_sentences"
default_in_dir = "/shared/dropbox/ctakes_conll/output/clinical/dc_summaries"
default_out_dir = "tokenized_sentences"

def func(x):
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

    if i%10 == 0:
        print ("\rfiles:%2d"%(i), end="    ")
        sys.stdout.flush()

	return  out_dir


if __name__ == '__main__':
	import argparse

	argparser = argparse.ArgumentParser()

	argparser.add_argument("-i", "--input_dir", dest="input_dir", type=str, default=default_in_dir, \
		help="input_dir (default={})".format(None))

	argparser.add_argument("-o", "--output_dir", dest="output_dir", type=str, default=default_out_dir, \
		help="output_dir (default={})".format(None))

	args = argparser.parse_args()

	in_dir = args.input_dir
	out_dir = args.output_dir

	idx_fs = [ (i, in_dir, out_dir, f) for i, f in enumerate(os.listdir(in_dir))]
	#ret_lst = [func(x) for x in idx_fs]
	ret_lst = multiprocess.Pool().imap_unordered(func, idx_fs)
	ret_lst = [r for r in ret_lst if r != None]
	print ("\ntotal files:", len(ret_lst), "\nsave to:", out_dir)
