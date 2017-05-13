import os
import sys
import re
import numpy as np
np.random.seed(3318)
import word2vec
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import brown
from nltk.tag import UnigramTagger
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from adjustText import adjust_text

# config
path = "data_wordvec"
k = 500

# text to phrase and generate model
if "--train" in sys.argv:
    word2vec.word2phrase(os.path.join(path, "all.txt"), os.path.join(path, "all-phrase.txt"), verbose = True)
    word2vec.word2vec(os.path.join(path, "all-phrase.txt"), os.path.join(path, "all.bin"), size = 50, verbose = True)

# load model
model = word2vec.load( os.path.join(path, "all.bin") )
print("model.vecab |", model.vocab)
print("model.vectors.shape |", model.vectors.shape)

# perform tsne
tsneModel = TSNE(n_components = 2, random_state = 0)
np.set_printoptions(suppress = True)
tsneVec = tsneModel.fit_transform(model.vectors[:k, :])
print("model.vocab[10] tsneVec[10] |", model.vocab[10], tsneVec[10])

# download nltk resource
if "--download" in sys.argv:
    nltk.download("averaged_perceptron_tagger")
    nltk.download("maxent_treebank_pos_tagger")
    nltk.download("punkt")
    nltk.download("brown")
tagger = UnigramTagger(brown.tagged_sents(categories = "news")[:5000])
print( "tagger.tag(\"think\") |", tagger.tag(["think"]) )

# plot
plt.rcParams["figure.figsize"] = (10, 8)
fig = plt.figure()
plt.clf()
x = []
y = []
label = []
for i in range(k):
    (voc, tag) = tagger.tag([ model.vocab[i] ])[0]
    if tag != None and ( not re.match("^(JJ|NNP|NN|NNS)$", tag) ):  continue
    if not re.match("^[a-zA-Z][a-zA-Z]+$", voc):    continue
    x.append(tsneVec[i, 0])
    y.append(tsneVec[i, 1])
    label.append(voc)
print("label[:10] |", label[:10])
plt.plot(x, y, "k.")
texts = [plt.text(x[i], y[i], label[i], fontsize = 8) for i in range( len(label) )]
adjust_text( texts, x, y, arrowprops = dict(arrowstyle = "-", color = "black", lw = .5) )
plt.savefig("wordvec.png", format = "png")

