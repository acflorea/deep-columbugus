from dbloader import getTextForDictionary, bugDicoToFullText, getBugDetails
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.linear_model import SGDClassifier

# retrieve descriptions from db
# bug_descs = getTextForDictionary(1000)

# fulltextdesc = bugDicoToFullText(bug_descs)

# print fulltextdesc

seed = 42

db = 'netbeansbugs'


# db = 'eclipsebugs'
# db = 'firefoxbugs_new'


#  ==========================================================
def fetchAndSaveDataframe(db):
    print("Database is %s" % db)
    print("Fetching data")

    bug_dataframe = getBugDetails(db)

    print("Data frame fetched, here's an extract")
    print bug_dataframe.head(10)

    print('Saving dataframe')
    bug_dataframe.to_csv("./%s.csv" % db, encoding='utf-8')

    return bug_dataframe


#  ==========================================================
def loadDataframe(db):
    bug_dataframe = pd.read_csv("./%s.csv" % db, encoding='utf-8', index_col=0)
    return bug_dataframe


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

bug_dataframe = loadDataframe(db)
print("=" * 80)
print("Dataframe loaded %s" % str(bug_dataframe.shape))

print("=" * 80)
print("Test/Train split")
train, test = train_test_split(bug_dataframe, train_size=0.8, random_state=seed)
y_train, y_test = train.assigned_to, test.assigned_to

print("Train is %s" % str(train.shape))
print("Test is %s" % str(test.shape))

print("=" * 80)
print("Vectorize")
labels = train.assigned_to
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')

X_train = vectorizer.fit_transform(train.text.astype('U'))
X_test = vectorizer.transform(test.text.astype('U'))

print("Vectorized!")
print("Train is %s" % str(X_train.shape))
print("Test is %s" % str(X_test.shape))

print("=" * 80)
print("Train model")

results = []

# results = benchmark(LinearSVC(loss='squared_hinge', penalty='l2',
#                               dual=False, tol=1e-3))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

print("=" * 80)
