# Wordembedding

### run w2v
using document text data set:

    python w2v -i txt.paths -m model_path
  
udpdate pre-trained model:

    w2v -i ../../docsim/docsim/data -m checkpt/ckpt-10000.model
  
Find similar words:

    python src/word2vec/helper.py -m checkpt/ckpt-10000.model -t "above-guideline agreements aggravated damages citation."
