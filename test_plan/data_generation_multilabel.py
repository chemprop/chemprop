import random
import pandas as pd

reader = open("./data/tox21.csv", "r")
lines = reader.readlines()
reader.close()
num_data = 300


writer = open("./data/tox21_multilabel1.csv", "w")
for line in lines[:100]:
    writer.write(line)

writer.close()

writer = open("./data/tox21_multilabel2.csv", "w")
writer.write(lines[0])
for line in lines[100:200]:
    writer.write(line)

writer.close()

writer = open("./data/tox21_multilabel3.csv", "w")
writer.write(lines[0])
for line in lines[200:300]:
    writer.write(line)

writer.close()
