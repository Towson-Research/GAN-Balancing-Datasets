#!/usr/bin/env python3

from mysql import SQLConnector

conn = SQLConnector()

attack = 'neptune'

json = conn.pull_best(attack)
json2 = conn.pull_best(attack, True)

print(json)
print(json2)

