import random


reader = open("./data/delaney.csv", "r")
lines = reader.readlines()
reader.close()
num_data = 300

lines[0] = "smiles,task1,task2,task3"

for i in range(1, num_data):
    index = lines[i].index(",")
    random_1 = 1.0*random.randint(0, 1000)/1000
    random_2 = 1.0*random.randint(0, 1000)/1000
    random_3 = 1.0*random.randint(0, 1000)/1000
    str_1 = str(random_1) if random.randint(0, 10) < 10 else ""
    str_2 = str(random_2) if random.randint(0, 10) < 10 else ""
    str_3 = str(random_3) if random.randint(0, 10) < 10 else ""
    lines[i] = lines[i][:index] + "," + str_1 + "," + str_2 + "," + str_3

writer = open("./data/delaney3class1.csv", "w")
for line in lines[:100]:
    writer.write(line + "\n")

writer.close()

writer = open("./data/delaney3class2.csv", "w")
writer.write(lines[0]+"\n")
for line in lines[100:200]:
    writer.write(line + "\n")

writer.close()

writer = open("./data/delaney3class3.csv", "w")
writer.write(lines[0]+"\n")
for line in lines[200:300]:
    writer.write(line + "\n")

writer.close()