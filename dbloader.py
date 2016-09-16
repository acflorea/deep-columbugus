import pymysql.cursors
import re

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='mysql',
                             db='netbeansbugs',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


def testConnection():
    try:
        with connection.cursor() as cursor:
            # Create a new record
            sql = "SELECT count(*) from BUGS"
            cursor.execute(sql)

        # connection is not autocommit by default. So you must commit to save
        # your changes.
        connection.commit()

        with connection.cursor() as cursor:
            # Read a single record
            sql = "SELECT `bug_id` FROM `bugs` WHERE `assigned_to`=%s"
            cursor.execute(sql, (3,))
            result = cursor.fetchone()
            print(result)
    finally:
        connection.close()

def getTextForDictionary(limit = None):
    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='mysql',
                                 db='netbeansbugs',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    descriptions = []

    try:
        with connection.cursor() as cursor:
            # fetch number of rows
            if limit:
                count = limit
            else:
                countSQL = "SELECT count(*) FROM longdescs"
                cursor.execute(countSQL)
                count  = cursor.fetchone()['count(*)']
            print "Found %s rows" % count

            # compute the necessary batches
            batchSize = 50000
            batches = count / batchSize + 1

            print "%d batches required" % batches

            for index in xrange(batches):
                print "Processing batch %d" % index
                # Load textual representation of bugs
                offset = index * batchSize
                sql = "SELECT thetext FROM longdescs limit %d, %d" % (offset, batchSize)
                cursor.execute(sql)
                for row in cursor:
                    text = row['thetext']
                    # we preserve everything alhanumeric - textual description
                    # we preserve the points and spaces
                    # any other character gets replaced by a space
                    text = re.sub('[^A-Za-z \.]+', ' ', text)
                    # remove duplicate spaces
                    text = re.sub('[ ]{2,}', ' ', text)
                    #  lower case
                    text = text.lower()
                    descriptions.append(text)

    finally:
        connection.close()

    # create a string with all the descriptions
    descriptions = "\n".join(descriptions)

    return descriptions