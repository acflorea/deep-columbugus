import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='mysql',
                             db='netbeansbugs',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

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