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

file = sys.argv[1]
print(file)
with open(file, 'rt') as data:
        reader = csv.reader(data)
        dataset.append(list(reader))

x = []
valueset = []

cubeNum = 0
for cube in dataset:
    for image in cube:
        latlon = image[2][1:-1].split(',')
        #print(latlon[0], latlon[1], latlon[0][5:10], latlon[1][5:10])
        if(int(latlon[0][5:10]) > 55000 and int(latlon[0][5:9]) < 80000 and int(latlon[1][5:9])>8145):
            value = [float(latlon[0][5:7]), float(latlon[1][5:9]), cubeNum, image[1]]
            valueset.append(value)
    cubeNum += 1


vals = {}
idx = 0
for i in valueset:
    if i[0] in vals:
        vals[i[0]]+= 1
    else:
        vals[i[0]] = 1

od = collections.OrderedDict(sorted(vals.items()))

invalidKeys = []

for i in od:
    if(od[i] < 35):
        invalidKeys.append(i)

for i in invalidKeys:
    od.pop(i, None)

map = []
for i in od:
    nl = []
    for value in valueset:
        if(value[0] == i):
            nl.append(value)

    sl = sorted(nl, key=lambda x:x[1])
    map.append(sl[0:33])

mgList = []
mgcols = int(len(map)/3)
mgrows = int(len(map[0])/3)
for i in range(mgcols):
    for j in range(mgrows):
        mg = []
        ridx = j*3
        cidx = i*3
        for k in range(3):
            for l in range(3):
                mg.append(map[cidx+k][ridx+l][-1])
        mgList.append([i, j, mg])

with open(sys.argv[2], 'w+') as of:
    wr = csv.writer(of, quoting=csv.QUOTE_MINIMAL)
    for line in mgList:
        wr.writerow(line)
