# Generic utility functions
from dbloader import getBugDetails
import pandas as pd
import numpy as np

# The database in use
#  ==========================================================
# db = 'netbeansbugs'
# db = 'eclipsebugs'
db = 'firefoxbugs_new'

# Global random generator seed
#  ==========================================================
seed = 42


#  ==========================================================
def loadDataframe(db):
    bug_dataframe = pd.read_csv("./%s.csv" % db, encoding='utf-8', index_col=0)
    return bug_dataframe


#  ==========================================================
def fetchAndSaveDataframe(db):
    print("Database is %s" % db)
    print("Fetching data")

    bug_dataframe = getBugDetails(db)

    print("Data frame fetched, here's an extract")
    print bug_dataframe.head(10)

    classes = bug_dataframe.assigned_to.unique().tolist()

    bug_dataframe['class'] = pd.Series(bug_dataframe.assigned_to.map(lambda x: classes.index(x)))

    print('Saving dataframe')
    bug_dataframe.to_csv("./%s.csv" % db, encoding='utf-8')

    return bug_dataframe


#  ==========================================================
def TFIDF(dataframe):
    from sklearn.cross_validation import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("=" * 80)
    print("Test/Train split")
    train, test = train_test_split(dataframe, train_size=0.8, random_state=seed)
    y_train, y_test = train['class'], test['class']

    print("Train is %s" % str(train.shape))
    print("Test is %s" % str(test.shape))

    print("=" * 80)
    print("Vectorize")
    labels = train['class']
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')

    X_train = vectorizer.fit_transform(train.text.astype('U'))
    X_test = vectorizer.transform(test.text.astype('U'))

    print("Vectorized!")
    print("Train is %s" % str(X_train.shape))
    print("Test is %s" % str(X_test.shape))

    return ((X_train, y_train), (X_test, y_test))


#  ==========================================================
def vectorizeDoc(dataframe):
    from sklearn.cross_validation import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("=" * 80)
    print("Test/Train split")
    train, test = train_test_split(dataframe, train_size=0.8, random_state=seed)
    y_train, y_test = train.assigned_to, test.assigned_to

    print("Train is %s" % str(train.shape))
    print("Test is %s" % str(test.shape))

    print("=" * 80)
    print("Vectorize")
    labels = train.assigned_to

    X_train = np.asarray([np.asarray(t) for t in train.embeddings.values])
    X_test = np.asarray([np.asarray(t) for t in test.embeddings.values])

    print("Vectorized!")
    print("Train is %s" % str(X_train.shape))
    print("Test is %s" % str(X_test.shape))

    return ((X_train, y_train), (X_test, y_test))
