from utils import loadDataframe, db

import os
import csv

import tensorflow as tf
from tensorflow.models.embedding.word2vec_optimized import Options, Word2Vec

# Build vocabulary file
dicoFile = "./%s_dico.csv" % db
dataFrame = loadDataframe(db)
dataFrame.to_csv(dicoFile,
                 columns=['text'], encoding='utf-8', index=False,
                 header=False)

# remove quotes
f = open(dicoFile, 'r')
text = f.read()
f.close()
text = text.replace("\"", "")
f = open(dicoFile, 'w')
f.write(text)
f.close()

"""Train a word2vec model."""

opts = Options()
opts.train_data = dicoFile
opts.save_path = "."
opts.eval_data = "questions-words.txt"
with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
        model = Word2Vec(opts, session)
    for _ in xrange(opts.epochs_to_train):
        model.train()  # Process one epoch
        model.eval()  # Eval analogies.
    # Perform a final save.
    model.saver.save(session, os.path.join(opts.save_path, "%s_model.ckpt" % db),
                     global_step=model.step)
