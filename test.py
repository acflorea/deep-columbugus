from __future__ import print_function
from utils import loadDataframe, db, seed, fetchAndSaveDataframe, TFIDF, vectorizeDoc

from time import time

from sklearn import metrics
from sklearn.svm import LinearSVC

import tensorflow as tf
from tensorflow.models.embedding.word2vec_optimized import Options, Word2Vec
from utils import db
import os

import numpy as np

from datetime import datetime, timedelta

import pandas as pd


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


fetchAndSaveDataframe(db)

def Doc2Vec(row, embeddings, word2id):
    # Retrieve word's embeddings
    ids = [word2id.get(word, -1) for word in row.text.split()]
    row_embeddings = [embeddings[id] for id in ids if id > -1]
    # Compute the average of the word representations (naive approach)
    # doc2vec = avg(word2voc)
    mean = np.asarray(row_embeddings).mean(axis=0)
    return mean.tolist()


bug_dataframe = loadDataframe(db)

# go back in time "delta" days
last_change = datetime.strptime(bug_dataframe['delta_ts'].max(), '%Y-%m-%d %H:%M:%S').date()
oneMonthBack = (last_change + timedelta(days=-30)).strftime('%Y-%m-%d')
twoMonthsBack = (last_change + timedelta(days=-60)).strftime('%Y-%m-%d')
threeMonthsBack = (last_change + timedelta(days=-90)).strftime('%Y-%m-%d')
break_at = (last_change + timedelta(days=-240)).strftime('%Y-%m-%d')

oneBackSeries = bug_dataframe[bug_dataframe['bug_when'] > oneMonthBack].groupby('assigned_to').filter(
    lambda x: len(x) > 3).assigned_to.unique()
twoBackSeries = bug_dataframe[
    (bug_dataframe['bug_when'] <= oneMonthBack) & (bug_dataframe['bug_when'] > twoMonthsBack)].groupby(
    'assigned_to').filter(lambda x: len(x) > 3).assigned_to.unique()
threeBackSeries = bug_dataframe[
    (bug_dataframe['bug_when'] <= twoMonthsBack) & (bug_dataframe['bug_when'] > threeMonthsBack)].groupby(
    'assigned_to').filter(lambda x: len(x) > 3).assigned_to.unique()

validAssignees = reduce(np.intersect1d, (oneBackSeries, twoBackSeries, threeBackSeries)).tolist()

counts = bug_dataframe[
    (bug_dataframe['bug_when'] > threeMonthsBack) & (bug_dataframe['assigned_to'].isin(validAssignees))].groupby(
    'assigned_to').assigned_to.value_counts()
threshold = 2 * counts.std() + counts.mean()

print('Level 1 threshold ', threshold)

filteredValidAssignees = bug_dataframe[
    (bug_dataframe['bug_when'] > threeMonthsBack) & (bug_dataframe['assigned_to'].isin(validAssignees))].groupby(
    'assigned_to').filter(lambda x: len(x) < threshold).assigned_to.unique().tolist()

# - repeat ?!?

filtered_bug_dataframe = bug_dataframe[
    (bug_dataframe['delta_ts'] > break_at) & (bug_dataframe['assigned_to'].isin(filteredValidAssignees))]
filtered_bug_dataframe = filtered_bug_dataframe.sort('delta_ts')[0:len(filtered_bug_dataframe.index) / 10 * 9]

counts = filtered_bug_dataframe \
    .groupby('assigned_to').assigned_to.value_counts()
threshold = 2 * counts.std() + counts.mean()

print('Level 2 threshold ', threshold)

filteredValidAssignees = filtered_bug_dataframe.groupby(
    'assigned_to').filter(lambda x: len(x) < threshold).assigned_to.unique().tolist()

# keep only "recent" data
filtered = bug_dataframe[
    (bug_dataframe['delta_ts'] > break_at) & (bug_dataframe['assigned_to'].isin(filteredValidAssignees))]

print('Saving dataframe')
filtered.to_csv("./%s_filtered.csv" % db, encoding='utf-8', header=False)
print('Saving dataframe')
bug_dataframe.to_csv("./%s.csv" % db, encoding='utf-8', header=False)

# get rid of nans
bug_dataframe = bug_dataframe.replace(np.nan, '', regex=True)
print("=" * 80)
print("Dataframe loaded %s" % str(bug_dataframe.shape))

# bug_dataframe = bug_dataframe.head(50000)

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

    bug_dataframe['embeddings'] = bug_dataframe.apply(Doc2Vec, args=(embeddings, model._word2id), axis=1)

    # print('Saving dataframe')
    bug_dataframe.to_csv("./%s_e.csv" % db, encoding='utf-8')

    print(bug_dataframe.head(10))

# tf-idf
# ((X_train, y_train), (X_test, y_test)) = TFIDF(bug_dataframe)
((X_train, y_train), (X_test, y_test)) = vectorizeDoc(bug_dataframe[bug_dataframe.embeddings > 0])

print("=" * 80)
print("Train model")

results = []

results = benchmark(LinearSVC(loss='squared_hinge', penalty='l2',
                              dual=False, tol=1e-3))

print("=" * 80)
