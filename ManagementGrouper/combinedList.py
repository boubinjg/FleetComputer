import sys
import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import collections

dataset = []
managementGroups = []

def getImage(imID):
    for d in dataset:
        if d[1] == imID:
            return d
    return []

f = sys.argv[1:]
for file in f:
    print(file)
    with open(file, 'rt') as data:
        reader = csv.reader(data)
        dataset.append(list(reader))


cubeNum = 0
newDS = []
count = 0
for cube in dataset:
    for image in cube:
        nimage = image
        nimage[1] = count
        newDS.append(nimage)
        count += 1

for image in newDS:
    print(image)

with open('newCube.csv', 'w+') as of:
	wr = csv.writer(of, quoting=csv.QUOTE_MINIMAL)
	for line in newDS:
		wr.writerow(line)
