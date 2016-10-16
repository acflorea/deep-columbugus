import pymysql.cursors
import re
import pandas as pd


# Retrieves the bugs textual information from `longdescs`
# The results are in the form of a dictionary {bug_id : (original_text, normalized_text)}
# ====================================================================================================
def getTextForDictionary(db, limit=None):
    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='mysql',
                                 db=db,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    bug_desc = {}

    try:
        with connection.cursor() as cursor:
            # fetch number of rows
            if limit:
                count = limit
            else:
                countSQL = "SELECT count(*) FROM longdescs"
                cursor.execute(countSQL)
                count = cursor.fetchone()['count(*)']
            print "Found %s rows" % count

            # compute the necessary batches
            batchSize = 50000
            batches = count / batchSize + 1

            print "%d batches required" % batches

            for index in xrange(batches):
                print "Processing batch %d" % index
                # Load textual representation of bugs
                offset = index * batchSize
                sql = "SELECT bug_id, thetext FROM longdescs limit %d, %d" % (offset, batchSize)
                cursor.execute(sql)
                for row in cursor:
                    bug_id = row['bug_id']
                    original_text = re.sub('\n', ' ', row['thetext'])
                    # we preserve everything alphanumeric - textual description
                    # we preserve the points and spaces
                    # any other character gets replaced by a space
                    text = re.sub('[^A-Za-z\.]+', ' ', original_text)
                    # remove duplicate spaces
                    text = re.sub('[ ]+', ' ', text)
                    # remove boundary points
                    text = re.sub(' \.', ' ', text)
                    text = re.sub('\. ', ' ', text)
                    # remove any single letter words
                    text = re.sub(' (. )+', ' ', text)
                    # remove successive points
                    text = re.sub('(\.){2,}', '', text)
                    #  lower case
                    text = text.lower()
                    if (bug_id in bug_desc):
                        newText = bug_desc[bug_id]['text'] + " " + text
                        newOriginalText = bug_desc[bug_id]['original_text'] + " " + original_text
                        bug_desc[bug_id] = {'text': newText, 'original_text': newOriginalText}
                    else:
                        bug_desc[bug_id] = {'text': text, 'original_text': original_text}


    finally:
        connection.close()

    return bug_desc


# Concatenates the dico value into a single text
# Depending on the normalized parameter, either the original or the normalized representation is used
# ====================================================================================================
def bugDicoToFullText(bug_descs, normalized=False):
    fulltextdesc = []
    for desc in bug_descs:
        if normalized:
            index = 0
        else:
            index = 1
        fulltextdesc.append(bug_descs[desc][index])
    return fulltextdesc


# Collects a bugs dataframe with the following columns
# ['bug_id', 'creation_ts', 'short_desc', 'bug_status', 'assigned_to', 'product_id',
# 'component_id', 'bug_severity', 'resolution', 'delta_ts']
#
# ====================================================================================================
def getBugDetails(db, bugIds=[], modifiedNoLaterThan='2000-01-01', withFullDescription=True):
    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='mysql',
                                 db=db,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:
            countSQL = "SELECT b.bug_id, b.creation_ts, b.short_desc, " \
                       "b.bug_status, ba.who as assigned_to, b.product_id, b.component_id, b.bug_severity, " \
                       "b.resolution, b.delta_ts " \
                       "FROM bugs b JOIN bugs_activity ba on b.bug_id = ba.bug_id " \
                       "and ba.added='FIXED' " \
                       "JOIN fielddefs fd on fd.id = ba.fieldid and fd.name = 'resolution' " \
                       "where " + resolutionFilter("b.") + \
                       "AND b.delta_ts > %s " + \
                       "AND b.bug_id not in (select d.dupe from duplicates d) " \
                       "ORDER by b.bug_id"

            cursor.execute(countSQL, (modifiedNoLaterThan))
            rows_list = []

            print "Reading bugs data"
            index = 0
            for row in cursor:
                if (index % 10000 == 0):
                    print "Current index is %d" % index
                data_dict = {col: row[col] for col in
                             ['bug_id', 'creation_ts', 'short_desc', 'bug_status', 'assigned_to', 'product_id',
                              'component_id', 'bug_severity', 'resolution', 'delta_ts']}
                rows_list.append(data_dict)
                index += 1

            bug_dataframe = pd.DataFrame(data=rows_list, index=[dict['bug_id'] for dict in rows_list])

    finally:
        connection.close()

    # Complete the data frame with full bugs descriptions
    if (withFullDescription):
        print("Join descriptions")
        descriptions = getTextForDictionary(db=db)
        desc_dataframe = pd.DataFrame(data=descriptions.values(), index=descriptions.keys())
        bug_dataframe = bug_dataframe.join(desc_dataframe)

    return bug_dataframe


#  ===============
def resolutionFilter(alias="b."):
    return "%sresolution = 'FIXED'" % alias
