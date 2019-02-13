#!/usr/bin/env python3
import pymysql.cursors

"""
CREATE TABLE stats (
    layer1 int,
    layer2 int,
    layer3 int,
    accuracy float(20)
);
"""

class SQLConnector(object):

    def __init__(self):
        """ Constructor """
        # Connect to the database
        self.connection = pymysql.connect(host='localhost',
                             user='ldeng',
                             password='cs*titanML',
                             db='results',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

    def write(self, l1, l2, l3, acc):
        try:
            with self.connection.cursor() as cursor:
                # Create a new record
                sql = "insert into stats (layer1, layer2, layer3, accuracy) values (%s, %s, %s, %s);"
                cursor.execute(sql, (str(l1), str(l2), str(l3), str(acc)))
                
            # connection is not autocommit by default. So you must commit to save
            # your changes.
            self.connection.commit()

        finally:
            pass

    def read(self, acc):
        try:
            with self.connection.cursor() as cursor:
                # Read a single record
                sql = "select * from stats where accuracy > %s"
                cursor.execute(sql, (str(acc)))
                result = cursor.fetchall()
                print(result)
        finally:
            pass

    def close(self):
        self.connection.close()


def main():
    """ Auto run main method """
    conn = SQLConnector()
    
    conn.read(70.4)


if __name__=="__main__":
    main()

