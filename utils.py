# Generic utility functions
from dbloader import getBugDetails
import pandas as pd

# The database in use
#  ==========================================================
db = 'netbeansbugs'
# db = 'eclipsebugs'
# db = 'firefoxbugs_new'

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

    print('Saving dataframe')
    bug_dataframe.to_csv("./%s.csv" % db, encoding='utf-8')

    return bug_dataframe
