from utils import loadDataframe, db

import os

import tensorflow as tf
from tensorflow.models.embedding.word2vec_optimized import Options, Word2Vec

# Build vocabulary file
dataFrame = loadDataframe(db).text.astype('U')
dataFrame.to_csv("./%s_dico.csv" % db, encoding='utf-8', index=False)

"""Train a word2vec model."""

opts = Options()
opts.train_data = "%s_dico.csv" % db
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
