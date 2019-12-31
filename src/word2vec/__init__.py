from __future__ import print_function
import sys
import os
import json
import gensim

from .get_files_multi import get_files
from .helper import save_word_count, save_w2v, get_vec, get_text_similar, split_dir
from .conll2tokens import tokenized_text_from_conll
from .w2v import read_and_preprocess, train

def get_latest(model_root, size):
    end_size_lastest = []
    for x in os.listdir(model_root):
        (e, d) = tuple(x.split('_')[0].split('d') )
        end_size_lastest += [(int(e), int(d), x)]

    if end_size_lastest == []:
        return (0, size, None)

    return sorted(end_size_lastest)[-1]

def get_file_paths(in_dir, out_file, ext='txt'):
    files = get_files(in_dir, ext)
    with open(out_file, 'w') as fd:
        fd.write('\n'.join(files))
    print (len(files), " Write to:", out_file)

def dir_to_sub_dir(one_level_dir, out_dir, op='cp', idx_start=0, step=10000, limit=2000000):
    print ("\n".join(split_dir(one_level_dir, out_dir, op=op, idx_start=idx_start, step=step, limit=limit)))

def save_wc(tokenized_text_root, out_file):
    print (save_word_count(tokenized_text_root, out_file))

def save_w2v_model(model_path, output_file, vocab_lst_path=""):
    print ("loading model:", model_path, " .....")
    model = gensim.models.Word2Vec.load(model_path)
    if vocab_lst_path != "":
        with open(vocab_lst_path, "r") as fd:
            vocab_lst = fd.read().strip().split('\n')
    else:
        vocab_lst = list(model.wv.vocab)
    print (save_w2v(model, vocab_lst, output_file))

def get_similar(model_path, text):
    print ("loading model:", model_path, " .....")
    model = gensim.models.Word2Vec.load(model_path)
    print(get_text_similar(model, text))

def _train(input_paths_file, checkpoint_dir, chunk_size, size, min_count, epochs, prep="segtok", resume=True, verbose=False):
    input_span = (0, 2000000)
    retrain = None
    if resume:
        (e, size, retrain) = get_latest(checkpoint_dir, size)
        input_span = (e, e + chunk_size)
    if retrain != None:
        retrain=os.path.join(checkpoint_dir, retrain, '%s.model'%('d'+'d'.join(retrain.split('d')[1:])))

    if prep == "simple":
        preprocessing_func = split_hyphen_segtok
    else:
        preprocessing_func = simply_split
    
    docs = read_and_preprocess(input_paths_file, preprocessing_func, checkpoint_dir, input_span=input_span, verbose=verbose)
    if docs["n_docs"] == 0:
        print ("ERROR: Empty dataset! input_span:", str(input_span))
        sys.exit(0)
        
    model_name = "d%d"%(size)
    save_model_dir = os.path.join(checkpoint_dir, str(input_span[1])+model_name)
    if not os.path.isdir(save_model_dir):
        os.system("mkdir %s"%save_model_dir)

    model = train(docs, size=size, min_count=min_count, \
        save_model=os.path.join(save_model_dir, '%s.model'%(model_name)), \
        retrain=retrain, \
        epochs=epochs, verbose=verbose)

    print (model)
    print ("model size:", getsize(model))

def trainer(input_paths_file, checkpoint_dir, chunk_size=1000, size=100, min_count=1, epochs=5, prep="segtok", resume=True, verbose=False):
    _train(input_paths_file, checkpoint_dir, chunk_size, size, min_count, epochs, prep, resume, verbose)

def trainer_config(config_path='config.json', verbose_new=None):
    with open(config_path) as fj:
        config = json.load(fj)
    input_path, model_root, chunk_size, epochs, resume, size, min_count, preprocessor, verbose = \
        config['input_path'], config['model_root'], config['chunk_size'], config['epochs'], \
        config['resume'], config['size'], config['min_count'], config['preprocessor'], config['verbose']
    if verbose_new == None:
        verbose = verbose_config
    else:
        verbose = verbose_new 
    _train(input_path, model_root, chunk_size, size, min_count, epochs, preprocessor, resume, verbose)    


