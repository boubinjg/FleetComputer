import sys
import csv
import subprocess

dataset = []
managementGroups = []

def getImage(imID):
    for d in dataset:
        if d[1] == imID:
            return d
    return []

def getNeighborhood(zone):
    image = getImage(zone)
    neighborhood = image[4:8]
    if(neighborhood == []):
        return []
    nimage = getImage(neighborhood[1])
    if(nimage == []):
        neighborhood.append([])
        neighborhood.append([])
    else:
        neighborhood.append(nimage[6])
        neighborhood.append(nimage[7])
    simage = getImage(neighborhood[2])
    if(simage == []):
        neighborhood.append([])
        neighborhood.append([])
    else:
        neighborhood.append(simage[6])
        neighborhood.append(simage[7])

    return neighborhood

def makeManagementGroup(idx):
    global managementGroups, dataset
    image = getImage(idx)
    neighbors = image[4:8]
    south = getImage(neighbors[1])
    neighborhood = getNeighborhood(south[5])

    return neighborhood

f = sys.argv[1]
with open(f, 'rt') as data:
    reader = csv.reader(data)
    dataset.append(list(reader))
    dataset = dataset[0]

idx = dataset[0][1]
for i in range(3):
    mg = makeManagementGroup(idx)
    print(mg)




'''with open(sys.argv[2], 'w+') as of:
	wr = csv.writer(of, quoting=csv.QUOTE_MINIMAL)
	for line in ndata:
		wr.writerow(line)'''
