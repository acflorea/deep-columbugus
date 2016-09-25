from __future__ import print_function
from utils import loadDataframe, db, seed, fetchAndSaveDataframe, TFIDF

from time import time

from sklearn import metrics
from sklearn.svm import LinearSVC

import tensorflow as tf
from tensorflow.models.embedding.word2vec_optimized import Options, Word2Vec
from utils import db
import os

import numpy as np


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print()

    print("classification report:")
    print(metrics.classification_report(y_test, pred))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


# fetchAndSaveDataframe(db)

def Doc2Vec(row, embeddings, word2id):
    # Retrieve word's embeddings
    ids = [word2id.get(word, -1) for word in row.text.split()]
    row_embeddings = [embeddings[id] for id in ids if id > -1]
    # Compute the average of the word representations (naive approach)
    # doc2vec = avg(word2voc)
    return (np.asarray(row_embeddings).mean(axis=0),)


bug_dataframe = loadDataframe(db)
# get rid of nans
bug_dataframe = bug_dataframe.replace(np.nan, '', regex=True)
print("=" * 80)
print("Dataframe loaded %s" % str(bug_dataframe.shape))

# load W2V model
opts = Options()
opts.train_data = "%s_dico.csv" % db
opts.save_path = "."
opts.eval_data = "questions-words.txt"
modelNo = "4349991"

with tf.Graph().as_default(), tf.Session() as session:
    model = Word2Vec(opts, session)
    model.saver.restore(session,
                        os.path.join(opts.save_path + "/%s_model.ckpt-%s" % (db, modelNo)))
    # get the embeddings
    embeddings = session.run(model._w_in, {})

    print("Enhancing dataframe")
    bug_dataframe['words'] = bug_dataframe.apply(Doc2Vec, args=(embeddings, model._word2id), axis=1)
    print(bug_dataframe.head(10))

# tf-idf
((X_train, y_train), (X_test, y_test)) = TFIDF(bug_dataframe)

print("=" * 80)
print("Train model")

results = []

results = benchmark(LinearSVC(loss='squared_hinge', penalty='l2',
                              dual=False, tol=1e-3))

print("=" * 80)
