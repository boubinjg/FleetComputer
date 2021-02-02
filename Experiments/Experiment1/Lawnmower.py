import sys
import csv
from sklearn.neighbors import NearestNeighbors
import numpy as np
import statistics
from collections import defaultdict
import copy
import os
from shutil import copyfile
from shutil import rmtree
import glob
import subprocess
import random
import math
import traceback
import time
from operator import itemgetter
import cv2

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

MAX_GROUP=15
ZONES_TO_VISIT=308
MAX_UTIL=0
U_GOAL = 0
X_DIM = 6
Y_DIM = 11
ACCURACY_GOAL = float(sys.argv[6])

imagedata = []
maxFeats = []
knndataSet = []
managementGroups = []
mgs = []
initialPosition=0


dirDict = {0: [[1,3],[3,1]],
           1: [[1,4],[2,0],[3,2]],
           2: [[1,5],[2,1]],
           3: [[0,0],[1,6],[2,4]],
           4: [[0,1],[1,7],[2,5],[3,3]],
           5: [[0,2],[1,8],[3,4]],
           6: [[0,3],[2,7]],
           7: [[0,4],[2,8],[3,6]],
           8: [[0,5],[3,7]],
            }


class Node:
        f = 0
        g = 0
        h = 0
        index = 0
        successor = 0
        def __init__(self,f,g,h,i,p,gi):
                self.f = f
                self.g = g
                self.h = h
                self.index = i
                self.parent = p
                self.gi = gi

def findLowestF(open):
        lowest = sys.maxsize
        lowestF = -1
        for node in open:
                if(node.f < lowest):
                        lowest = node.f
                        lowestF = node

        return lowestF


def gpsDist(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

def readInData():
    global imagedata, knndataSet, managementGroups
    with open(sys.argv[1], 'rt', encoding="utf-8") as data:
        reader = csv.reader(data)
        imagedata.append(list(reader))

    with open(sys.argv[2], 'rt', encoding="utf-8") as data:
        reader = csv.reader(data)
        managementGroups.append(list(reader))

    imagedata = imagedata[0]
    managementGroups = managementGroups[0]

    knnds = np.genfromtxt(sys.argv[3],delimiter=',')
    knndataSet.append(knnds)

def getFeatureMaximums():
    global imagedata, maxfeats

    nfeats = len(getFeatures(imagedata[0]))-1
    maxfeats = np.zeros(nfeats)

    for i in imagedata:
        fv = getFeatures(i)
        for f in range(nfeats):
            curFeat = abs(fv[f])
            if(curFeat > maxfeats[f]):
                maxfeats[f] = curFeat

def findSamples(points):
    global imagedata
    samples = []
    for p in points:
        samples.append(imagedata[p])
    return samples

def getFeatures(image):
    feat = image[12][1:-1].split(',')
    knnfeat = ""
    for f in feat:
        knnfeat += str(f.split('=')[1]) + ','
    return np.fromstring(knnfeat[:-1], sep=',')

def findGIList(n, knndata):
    return knndata[n][13:17]

def findFeatures(n):
    return knndata[n][0:12]

def findDirection(ug, image):
    direct = -1
    nextPic = []
    for i in range(0,4):
        if ug[i] > direct:
            direct = ug[i]
            nextPic = image[i+4]
    return nextPic

def getErr(gi, fieldmap):
    mean = statistics.mean(fieldmap)
    median = statistics.median(fieldmap)
    rangeMean = (min(fieldmap)+max(fieldmap))/2
    if(len(fieldmap) > 1):
        CImin = mean - statistics.stdev(fieldmap)*1.96
        CImax = mean + statistics.stdev(fieldmap)*1.96
    else:
        CImin = float(gi)
        CImax = float(gi)

    return [mean, median, rangeMean, CImin, CImax]

def errChange(gi, inmap, oldErr):
    fieldmap = inmap[:]
    fieldmap.append(float(gi))

    stats = getErr(gi, fieldmap)
    newErr = max(stats) - min(stats)
    return oldErr - newErr #maximize this value

def findGain(image, knndata):
    query = getFeatures(image)

    nbors = NearestNeighbors(int(sys.argv[5]))
    nbors.fit(knndata[:,0:12])

    knn = nbors.kneighbors([query[0:12]])
    knn = knn[1][0]
    dirs = [[0,0],[0,0],[0,0],[0,0]]
    for n in knn:
        ug = findGIList(n, knndata)
        for d in range(0,4):
            dirs[d][0] += 1
            dirs[d][1] += ug[d]

    mapGain = [d[1] / (d[0]-.00001) for d in dirs]

    gainMap = []
    for i in range (0,4):
        gainMap.append(mapGain[i])

    return gainMap

def getImage(imID):
    for d in imagedata:
        if d[1] == imID:
            return d
    return []


def writeUtilAst(visited, imdata, profile):
    try:
        rmtree('tmp')
    except Exception as e:
        print(e)

    os.mkdir("tmp")

    for im in visited:
        line = getImage(im)
        imageName = line[3][17:].split('/')[-1]
        gi = float(line[12].split(',')[10].split('=')[1][:-1])

        testVisited = copy.deepcopy(visited)
        testVisited.append(im)
        ret = []

        features = getFeatures(line)

        ret = features.tolist()

        for neighbor in line[4:8]:
            if(neighbor == '[]'):
                curImg = findClosest(image)
            else:
                curImg = getImage(neighbor)
            energy = findEnergy(curImg, testVisited, profile, gi)
            ret.append(min(energy))
        fname = imageName.split('.')[0]
        with open('tmp/'+fname+'.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(ret)


def writeUtil(visited, imdata):
    for im in visited:
        for line in imdata:
            if(im == line[1]):
                imageName = line[3][17:].split("/")[-1]
                util = float(line[12].split(',')[12].split('=')[1][:-1])
                utilList = []
                for i in range(4):
                    im2 = line[i+4]
                    if(im2 in visited):
                        im2Line = getImage(im2)
                        im2util = float(im2Line[12].split(',')[12].split('=')[1][:-1])
                        utilList.append(im2util/util)
                    else:
                        utilList.append('1')
                fname = imageName.split('.')[0]
                with open('tmp/'+fname+'.csv','w') as f:
                    writer = csv.writer(f)
                    writer.writerow(utilList)

def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def init_emptyzones(map, visible, search):
    empty = []
    shape = map.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            nz = [i,j]
            if(nz not in visible and nz not in search):
                nz.append(empty)

    return empty

def init_searchzones(map, visited):
    search = []
    for zone in visited:
        neighborhood = getNeighborhood(zone, map)
        for i in neighborhood:
            if i not in search and i not in visited and i != '[]':
                search.append(i)
    return search

def getNeighborhood(zone, map):
    shape = map.shape
    neighbors = []
    for i in range(-1,2):
        for j in range(-1,2):
            nz = [zone[0]+i, zone[1]+j]
            if(nz[0] < 0 or nz[0] >= shape[0] or nz[1] < 0 or nz[1] >= shape[1]):
                continue
            else:
                neighbors.append(nz)

    return neighbors

def find_maxzone(search, visible, map):
    max_value = 0
    max_zone = ''
    max_neighbor = []
    for idx in search:
        count = 0
        closest = []
        nbh = getNeighborhood(idx, map)
        for n in nbh:
            if n in visible:
                count += 1
                closest.append(n)
        if count > max_value:
            max_zone = idx
            max_value = count
            max_neighbor=closest
    return max_zone, max_neighbor

def getAverageGEx(imagedata):
    giSum = 0
    for img in imagedata:
        gi = float(img[12].split(',')[10].split('=')[1])
        giSum += gi
    return giSum / len(imagedata)

def getAverageGExMGS(mgs):
    giSum = 0
    count = 0
    for group in mgs:
        imgs = group[2][1:-1].split(',')
        for image in imgs:
            imgIdx = image.strip()[1:-1]
            imageVal = getImage(imgIdx)
            gi = getGI(imageVal)
            giSum += gi
            count += 1
    return giSum / count


def fillVisited(kernelmap, visited, imagedata):
    for zone in visited:
        image = getImage(zone)
        gt = float(image[12].split(',')[10].split('=')[1])
        agex = getAverageGEx(imagedata)
        pred = 1
        if(gt > agex*.8):
            pred = 0
        idx = int(zone)-1
        kernelmap[idx] = pred
    return kernelmap

def update_map(max_zone, empty_zones, visible_zones, search_zones):
    if len(empty_zones) != 0:
        for neighbor in getNeighborhood(max_zone):
            if neighbor in empty_zones:
                empty_zones.remove(neighbor)
                search_zones.append(neighbor)
    visible_zones.append(max_zone)
    search_zones.remove(max_zone)

def getcoords(zones):
    visited = []
    for group in zones:
        row = group[0]//X_DIM
        col = group[0]%X_DIM

        for zone in group[1]:
            idxr = row*3+zone//3
            idxc = col*3+zone%3
            visited.append([idxc, idxr])

    return visited

def extrapolate(GTmap, visited, imagedata):

    visible_zones = getcoords(visited)
    empty_zones = []

    search_zones = init_searchzones(GTmap, visible_zones)
    empty_zones = init_emptyzones(GTmap, visible_zones, search_zones)

    while len(search_zones) != 0:
        max_zone, max_neighbor = find_maxzone(search_zones, visible_zones, GTmap)
        total_pred = 0.0
        pred_count = 0
        for key in max_neighbor:
            if(key in visible_zones):
                pred = GTmap[key[0]][key[1]]
                total_pred += pred
                pred_count += 1
        final_pred = total_pred / pred_count
        GTmap[max_zone[0]][max_zone[1]] = final_pred
        #print(final_pred)

        update_map(max_zone, empty_zones, visible_zones, search_zones)
        #print(['visited='+str(len(visible_zones)),'empty='+str(len(empty_zones)),'search='+str(len(search_zones))])
    return GTmap

def checkAcc(map, imagedata, mgs):
    global X_DIM
    right = 0
    avgex = getAverageGExMGS(mgs)
    shape = map.shape

    for g in range(len(mgs)):
        group = mgs[g]
        imgs = group[2][1:-1].split(',')
        zone = 0


        row = int(group[1])
        col = int(group[0])
        groupnum = row*X_DIM+col
        #print(row,col,groupnum)

        for imgNum in range(len(imgs)):
            imgIdx = imgs[imgNum].strip()[1:-1]
            imageVal = getImage(imgIdx)
            gi = getGI(imageVal)

            pred = 0

            if(gi>1*avgex):
                pred = 0
            elif(gi>.9*avgex):
                pred = 1
            elif(gi>.8*avgex):
                pred = 2
            else:
                pred = 3

            idxr = row*3+zone//3
            idxc = col*3+zone%3

            #print(row, col, zone, idxr, idxc, imgIdx, gi, avgex)
            #print(group)
            if(map[idxc][idxr] > 2.5 and pred ==3):
                right += 1
            elif(map[idxc][idxr] > 1.5 and pred == 2):
                right += 1
            elif(map[idxc][idxr] > .5 and pred == 1):
                right += 1
            elif(map[idxc][idxr] < .5 and pred == 0):
                right += 1

            zone += 1

    #print('Accuracy: '+str(right/(shape[0]*shape[1])))
    showGTmap(map)
    return right/(shape[0]*shape[1])


def getGroup(mg):
    group = mg[-1]
    group = group[1:-1].split(',')
    formattedGroup = []
    for i in group:
        i = i.strip()[1:-1]
        formattedGroup.append(i)
    return formattedGroup

def getGI(img):
    gi = float(img[12].split(',')[10].split('=')[1])
    return gi

def getUtil(img):
    util = float(img[12].split(',')[12].split('=')[1][:-1])
    return util

def getNeighbors(row, col):
    global X_DIM, Y_DIM
    neighbors = []
    if(row-1 >= 0):
        neighbors.append([row-1, col])
    else:
        neighbors.append([])
    if(row+1 < 3*Y_DIM):
        neighbors.append([row+1, col])
    else:
        neighbors.append([])
    if(col+1 < 3*X_DIM):
        neighbors.append([row, col+1])
    else:
        neighbors.append([])
    if(col-1 >= 0):
        neighbors.append([row, col-1])
    else:
        neighbors.append([])


    return neighbors

def findNextImg(gm, row, col, visited, visitedNeighbors):
    neighbors = getNeighbors(row,col)

    maxdir = 0
    dirval = 0
    for n in range(len(neighbors)):
        if(gm[n] > dirval and neighbors[n] not in visited and neighbors[n] != []):
            maxdir = neighbors[n]
            dirval = gm[n]

    for n in range(len(neighbors)):
        if(neighbors[n] not in visited and neighbors[n] != [] and neighbors[n] not in visitedNeighbors and neighbors[n] != maxdir):
            visitedNeighbors.append(neighbors[n])

    if(dirval == 0):
        for n in visitedNeighbors:
            if(n not in visited):
                return n[0], n[1], visitedNeighbors
        print('Wow, you did it!, you created an error!')
        exit()

    return maxdir[0], maxdir[1], visitedNeighbors

def getImgRC(row, col, mgs, imagedata):
    global X_DIM
    groupnum = (row//3)*X_DIM+col//3

    zonenum = (row-(row//3)*3)*3 + (col-(col//3)*3)

    group = mgs[groupnum][2][1:-1].split(',')

    pic = group[zonenum].strip()
    img = getImage(pic[1:-1])
    return img


def explore(mg, imagedata, pos):
    global knndataSet, ZONES_TO_VISIT, MAX_UTIL, X_DIM, Y_DIM

    visited = []
    visitedNeighbors = []
    util = 0

    row = 0
    col = 0

    x_bound = X_DIM*3
    y_bound = Y_DIM*3

    mod = 1

    while(len(visited) < ZONES_TO_VISIT):
        visited.append([row,col])
        #Returns NSEW utility gain
        if((row < y_bound-1 and mod == 1) or (row>0 and mod == -1)):
            row = row+mod
        elif((row == y_bound-1 and mod == 1) or (row==0 and mod == -1)):
            mod *= -1
            col = col+1
        else:
            print('Huh?')

    return visited, pos

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def showGTmap(map):
    c = mcolors.ColorConverter().to_rgb
    heatMap = make_colormap([c('limegreen'), 0.25, c('yellow'), 0.5, c('orange'), 0.75, c('red')])
    fig = plt.figure()
    f = plt.imshow(map, cmap=heatMap)
    fig.savefig('Lawnmower.png', dpi=fig.dpi)

def getNextGroup(groupnum):
    global MAX_GROUP, X_DIM, Y_DIM

    row = groupnum//X_DIM
    if row%2 == 0:
        if (groupnum+1)%X_DIM == 0:
            return groupnum+X_DIM
        else:
            return groupnum+1
    else:
        if groupnum%X_DIM == 0:
            return groupnum+X_DIM
        else:
            return groupnum-1

    return -1

def convertToGroups(groupValues):
    global X_DIM, Y_DIM
    finalMap = []
    for i in range(X_DIM*Y_DIM):
        finalMap.append([i,[]])

    for i in groupValues:
        group = i[0]//3 * X_DIM + i[1]//3
        zonenum = (i[0]-(i[0]//3)*3)*3 + (i[1]-(i[1]//3)*3)
        finalMap[group][1].append(zonenum)

    return finalMap

def simulateMission(mgs, imagedata, pos):
    groupNum = 0
    visitedGroups = []
    finalMap = []
    #Explore all management zones!
    groupValues, npos = explore(mgs, imagedata, pos)
    finalMap = convertToGroups(groupValues)

    #print(visitedGroups)
    return finalMap

def buildGTMap(mgs, imagedata, visitedZones):
    GTmap = np.zeros((X_DIM*3, Y_DIM*3))

    avgex = getAverageGExMGS(mgs)
    for i in range(len(visitedZones)):
        group = visitedZones[i][0]
        row = group//X_DIM
        col = group%X_DIM

        for zone in visitedZones[i][1]:
            idxr = row*3+zone//3
            idxc = col*3+zone%3

            imgIdx = mgs[group][2][1:-1].split(',')[zone].strip()[1:-1]

            image = getImage(imgIdx)
            gi = getGI(image)

            #print(row, col, zone, idxr, idxc, imgIdx, gi, avgex)
            #print(mgs[group])

            if(gi>1*avgex):
                GTmap[idxc][idxr] = 0
            elif(gi>.9*avgex):
                GTmap[idxc][idxr] = 1
            elif(gi>0.8*avgex):
                GTmap[idxc][idxr] = 2
            else:
                GTmap[idxc][idxr] = 3

    return GTmap

def getNumVisited(zones):
    visited = 0
    for group in zones:
        visited += len(group[1])
    return visited

def runSim():
    global mgs, imagedata, initialPosition
    visitedZones = simulateMission(mgs, imagedata, initialPosition)
    GTMap = buildGTMap(mgs, imagedata, visitedZones)
    fullMap = extrapolate(GTMap, visitedZones, imagedata)
    acc = checkAcc(fullMap, imagedata, mgs)

    numVisited = getNumVisited(visitedZones)


    return acc, numVisited

def buildParamSet(featureNum, mgSize):
    params = defaultdict(dict)
    epsilon = 0.00001
    for i in range(featureNum):
        name = 'feature'+str(i)
        params[name] = hp.uniform(name, epsilon, 1)

    params['Vi'] = hp.uniform('Vi', 1, 9)
    params['Ui'] = hp.uniform('Ui', epsilon, 9) #How should we limit this value?
    return params

def rebuildUtility(params):
    global imageData, maxfeats

    weights = []
    nfeats = getFeatures(imagedata[0])
    for p in range(len(nfeats)-1):
        name = 'feature'+str(p)
        weight =  params[name]
        weights.append(weight)

    for i in range(len(imagedata)):
        fv = getFeatures(imagedata[i])
        utility = 0
        for f in range(len(weights)):
            utility += (abs(float(fv[f])) * weights[f])/maxfeats[f]

        utility /= (len(nfeats)-1)

        flist = imagedata[i][12][1:-1].split(',')
        #flist[12] = ' Utility='+str(utility)
        entry = '['
        for f in range(len(flist)-1):
            prefix = ''
            if(f != 0):
                prefix = ''
            entry += prefix+flist[f]+','
        entry += ' Utility='+str(utility)+']'
        imagedata[i][12] = entry

def objective_function(params):
    global ZONES_TO_VISIT, MAX_UTIL
    Vi = params['Vi']
    Ui = params['Ui']
    ZONES_TO_VISIT = int(Vi)
    MAX_UTIL = float(Ui)

    rebuildUtility(params)

    accuracy, numVisited = runSim(params)

    if(accuracy > ACCURACY_GOAL):
        loss = numVisited
    else:
        loss = sys.maxsize

    return loss

if __name__ == '__main__':

    readInData()
    getFeatureMaximums()
    imagedata = imagedata[:]
    num_eval = float(sys.argv[4])

    missions = 0

    count = 0
    means = []
    print("Start")

    #Number of Management Zones to explore
    initialPosition = 0

    #mgs = managementGroups[0:3]+ managementGroups[11:14]+ managementGroups[22:25]+ managementGroups[33:36]+ managementGroups[44:47]
    #mgs = managementGroups[0:55:11] + managementGroups[1:56:11] + managementGroups[2:57:11] + managementGroups[3:58:11] + managementGroups[4:59:11] + managementGroups[5:60:11] + managementGroups[6:61:11] + managementGroups[7:62:11] + managementGroups[8:63:11] + managementGroups[9:64:11] + managementGroups[10:65:11]
    mgs = managementGroups[0:65:11] + managementGroups[1:66:11] + managementGroups[2:67:11] + managementGroups[3:68:11] + managementGroups[4:69:11] + managementGroups[5:70:11] + managementGroups[6:71:11] + managementGroups[7:72:11] + managementGroups[8:73:11] + managementGroups[9:74:11] + managementGroups[10:75:11]

    thold = 0.7

    for i in range(42,43):
        ZONES_TO_VISIT = 10*i

        accuracy,numVisited = runSim()

        #print(numVisited, accuracy)
        if(accuracy>thold):
            print([thold,numVisited,accuracy])
            thold+=.05


    print('Best Performance: ')
    print(accuracy, numVisited)
