"""

    python src/word2vec/compare_models.py -i text.txt -m /shared/dropbox/ctakes_conll/w2v/pmc_d300.txt,/shared/dropbox/ctakes_conll/w2v/ctakes_d300.txt

"""
from gensim.models import KeyedVectors
import string


def get_similar(path, text):
    print("loading model from: %s ....."%path)
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=False)
    
    for p in string.punctuation:
        text = text.replace(p, " "+p+" ")
    text = text.replace("  ", " ").replace("  ", " ")
    tokens = text.split()

    ret_str = ''
    for tok in tokens:
        ret_str += "%15s :"%(tok)
        try:
            similars = [w for (w, v) in word_vectors.most_similar(positive=tok,topn=5)]
            ret_str += ','.join([" %s"%(tok) for tok in similars]) + '\n'
        except Exception as e:
            ret_str += " ERROR: %s"%e + '\n'

    return ret_str


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument("-i", "--input_file", dest="input_file", type=str, default=None, \
        help="input_file (default={})".format(None))

    argparser.add_argument("-m", "--model_vec_paths", dest="model_vec_paths", type=str, default=None, \
        help="model_vec_paths (default={})".format(None))

    args = argparser.parse_args()

    model_paths = args.model_vec_paths.split(",")
    with open(args.input_file, "r") as fd:
        text = fd.read()

    for path in model_paths:
        print(get_similar(path, text))
