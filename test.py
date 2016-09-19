from dbloader import getTextForDictionary, bugDicoToFullText, getBugDetails
import pandas as pd

# retrieve descriptions from db
# bug_descs = getTextForDictionary(1000)

# fulltextdesc = bugDicoToFullText(bug_descs)

# print fulltextdesc

# db = 'netbeansbugs'
# db = 'eclipsebugs'
db = 'firefoxbugs_new'


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


def loadDataframe(db):
    bug_dataframe = pd.read_csv("./%s.csv" % db, encoding='utf-8', index_col=0)
    return bug_dataframe


# fetchAndSaveDataframe(db)
bug_dataframe = loadDataframe(db)
print(bug_dataframe.shape)
print(bug_dataframe.head())
