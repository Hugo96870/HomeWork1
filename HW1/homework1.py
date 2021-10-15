# Grupo 117 Aprendizagem HomeWork 1
# Bernardo Castico ist196845
# Hugo Rita ist196870


import math
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from operator import itemgetter

Res = KFold(n_splits=10, random_state=117, shuffle=True)

def getDataToMatrix(lines):
    realLines = []
    data = []
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
            data += [realLines[i]]
    return data

def knn(data, i, train, k, numberWrongs, numberRights):
    lowerIndexes = []
    kLower = []
    for j in train:
        distance = euclidianDistance(data[i], data[j])
        if len(kLower) < k:
            kLower += [[distance, j]]
        else:
            higher = 0
            for q in range(1,k):
                if kLower[q][0] > kLower[higher][0]:
                    higher = q
            if kLower[higher][0] > distance:
                kLower[higher] = [distance,j]
    for q in range(len(kLower)):
        lowerIndexes += [data[kLower[q][1]][-1]]
    mode = statistics.mode(lowerIndexes)
    if mode == data[i][-1]:
        numberRights += 1
    else:
        numberWrongs += 1
    return [numberRights, numberWrongs]

def knn2(data, i, k, numberWrongs, numberRights):
    kLower = []
    lowerIndexes = []
    for j in range(len(data)):
        if j != i:
            distance = euclidianDistance(data[i], data[j])
            if len(kLower) < k:
                kLower += [[distance, j]]
            else:
                higher = 0
                for q in range(1, k):
                    if kLower[q][0] > kLower[higher][0]:
                        higher = q
                if kLower[higher][0] >= distance:
                    kLower[higher] = [distance, j]
    for q in range(len(kLower)):
        lowerIndexes += [data[kLower[q][1]][-1]]
    mode = statistics.mode(lowerIndexes)
    if mode == data[i][-1]:
        numberRights += 1
    else:
        numberWrongs += 1
    return[numberRights, numberWrongs]


def kFold1(data, k):
    results = [0,0]
    for train, test in Res.split(data):
        for i in test:
            resultsAux = knn(data, i, train, k, 0, 0)
            results[0] += resultsAux[0]
            results[1] += resultsAux[1]
    return results

def kFold2(data, k):
    results = [0,0]
    for i in range(len(data)):
        resultsAux = knn2(data, i, k, 0, 0)
        results[0] += resultsAux[0]
        results[1] += resultsAux[1]
    return results


def euclidianDistance(ponto1, ponto2):
    distance = 0
    for i in range(0, 9):
        distance += (ponto1[i] - ponto2[i]) ** 2
    distance = math.sqrt(distance)
    return distance

def main():
    res = []
    k = eval(input("k: "))
    with open("HW1.txt") as f:
        lines = f.readlines()
    for line in lines:
        tmp = line.split(',')
        res.append(tmp)
    data = getDataToMatrix(res)

    results1 = kFold1(data, k)

    accuracy1 = (results1[0]/(results1[0] + results1[1]))

    print("accuracy test: " + str(accuracy1))

    results2 = kFold2(data, k)

    accuracy2 = (results2[0]/(results2[0] + results2[1]))

    print("accuracy train: " + str(accuracy2))

main()
