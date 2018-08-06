"""
get_files_multi.py

	python get_files_multi.py
"""
from __future__ import unicode_literals
from __future__ import print_function
import os,sys,time
from multiprocessing import Pool, freeze_support

default_root = '/shared/data/PMC/txt'
default_out = 'txt.exist.paths'

def func(child):
	(dname, ext) = child
	files = []
	for root, directories, filenames in os.walk(dname):
		for filename in [f for f in filenames if f.endswith(ext)]:
			files += [os.path.join(root,filename) ]
	return files

def get_files(multi_dir, ext='txt'):
	try:
		children = [ (os.path.join(multi_dir, c), ext) for c in next(os.walk(multi_dir))[1]]
	except StopIteration as e:
		print ("ERROR:StopIteration")
		return []

	freeze_support()
	pool = Pool()

	files = []
	for lst in pool.map(func, children):
		files += lst

	return files


if __name__=="__main__":

	import argparse
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-i', '--input_path', dest='input_path', type=str, default=default_root,\
		help="input root")
	argparser.add_argument('-o', '--output_path', dest='output_path', type=str, default=default_out,\
		help="input root")
	argparser.add_argument('-e', '--ext', dest='ext', type=str, default='txt',\
		help="extension")
	args = argparser.parse_args()

	input_path = args.input_path
	ext = args.ext
	if ext == 'nxml':
		(ext, multi_dir, outpath) = ('nxml', input_path, 'nxml_exist.txt')
	elif ext == 'txt':
		(ext, multi_dir, outpath) = ('txt', input_path, args.output_path)
	elif ext == 'pkl':
		(ext, multi_dir, outpath) = ('pkl', input_path, 'pkl_exist.txt')
	else:
		print ('ERROR: -e'); sys.exit(0)

	t0 = time.time();print(multi_dir, ext)
	files = get_files(multi_dir, ext)

	print ('\n'.join(files[:3]))
	print ()
	print (time.time() - t0)
	with open(outpath, 'w') as fd:
		fd.write('\n'.join(files))
	print (len(files), " Write to:", outpath)
