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
                             port=3306,
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

    def read(self, acc = 0):
        try:
            with self.connection.cursor() as cursor:
                # Read a single record
                sql = "select * from stats where accuracy > %s"
                cursor.execute(sql, (str(acc)))
                result = cursor.fetchall()
                return result
        finally:
            pass


    def read_to_csv(self, filename = "stats.csv", acc = 0):
        results = self.read(acc)  # list of dicts

        string = ""
        for i in range(len(results)):
            dict = results[i]
            for key, value in dict.items():
                string += str(value)
                if key != "accuracy":
                    string += ","
            string += "\n"

        f = open(filename, "w")
        f.write(string)

    def clear(self):
        try:
            with self.connection.cursor() as cursor:
                # Create a new record
                sql = "drop table stats;"
                sql2 = "CREATE TABLE stats (layer1 int, layer2 int, layer3 int, accuracy float(20));"
                cursor.execute(sql)
                cursor.execute(sql2)
                
            # connection is not autocommit by default. So you must commit to save
            # your changes.
            self.connection.commit()

        finally:
            pass

    def close(self):
        self.connection.close()


def main():
    """ Auto run main method """
    conn = SQLConnector()

    conn.clear()
    #conn.write(1, 2, 3, 23.11)
    #print(conn.read())
    #conn.read_to_csv("abc.csv", 20)


if __name__=="__main__":
    main()

