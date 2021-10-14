# Grupo 117 Aprendizagem HomeWork 1
#Bernardo Castiço ist196845
#Hugo Rita ist196870


import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

k = 10
Res = KFold(n_splits = k, random_state = 117, shuffle = True)

def getDataToMatrix(lines):
    realLines = []
    realLines1 = []
    toDelete = []
    for i in range(len(lines)):
        if i > 11:
            realLines += [lines[i]]
    for i in range(len(realLines)):
        for j in range(len(realLines[i])):
            if realLines[i][j] == "benign\n":
                realLines[i][j] = 1
            elif realLines[i][j] == "malignant\n":
                realLines[i][j] = 0
            elif realLines[i][j] == '?':
                toDelete += [i]
            else:
                realLines[i][j] = int(realLines[i][j])
    for i in range(len(realLines)):
        if i not in toDelete:
            realLines1 += [realLines[i]]
    return realLines1

def euclidianDistance(ponto1, ponto2):
    distance = 0
    for i in range(0,9):
        distance += (ponto1[i] - ponto2[i])**2
    distance = math.sqrt(distance)
    return distance

def getLowerDistance(k, array):
    lower = []
    n = len(array)

    for i in range(n-1): #sort the list to get the lower elements
        for j in range(0, n-i-1):
            if array[j] > array[j + 1] :
                array[j], array[j + 1] = array[j + 1], array[j]
    
    for i in range(k):
        lower += array[i]
    return lower

def knn():
    a = 1

def main():
    res = []
    with open("HW1.txt") as f:
        lines = f.readlines()
    for line in lines:
        tmp = line.split(',')
        res.append(tmp)
    getDataToMatrix(res)


if __name__ == '__main__':
    main()