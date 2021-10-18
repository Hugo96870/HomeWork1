# Grupo 117 Aprendizagem HomeWork 1
# Bernardo Castico ist196845
# Hugo Rita ist196870


import math
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

Res = KFold(n_splits=10, random_state=117, shuffle=True)

#Atributes
atributes = ["Clump Thickness", "Cell Size Uniformity", "Cell Shape Uniformity", "Marginal Adhesion", "Single Epi Cell Size",
"Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]

#Get data

def getVectors(vector, index):
    res = []
    for i in vector:
        res.append(i[index])
    return res

def divideData(data):
    benign = []
    malignant = []

    for i in data:
        if i[-1] == 1:
            benign += [i]
        else:
            malignant += [i]

    return [benign, malignant]

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

#Exercise 6

def knn(data, i, train, k, numberWrongs, numberRights):
    lowerIndexes = []
    kLower = []  #Stores the k lower euclidian distances in comparison to data[i]
    for j in train:
        distance = euclidianDistance(data[i], data[j])
        if len(kLower) < k:
            kLower += [[distance, j]]
        else:
            higher = 0            #Assume that the first element is the higher
            for q in range(1,k):
                if kLower[q][0] > kLower[higher][0]:
                    higher = q
            if kLower[higher][0] > distance:
                kLower[higher] = [distance,j]
    for q in range(len(kLower)):
        lowerIndexes += [data[kLower[q][1]][-1]]
    mode = statistics.mode(lowerIndexes)               #Calculates the mode using
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
    resultsAux1 = [0,0]
    accuracies1 = []
    resultsAux2 = [0,0]
    accuracies2 = []
    for train, test in Res.split(data):
        for i in test:
            resultsAux = knn(data, i, train, k, 0, 0)
            results[0] += resultsAux[0]
            results[1] += resultsAux[1]
            resultsAux1[0] += resultsAux[0]
            resultsAux1[1] += resultsAux[1]
        accuracies1 += [resultsAux1[0]/(resultsAux1[1]+resultsAux1[0])]
        resultsAux1[0] = 0
        resultsAux1[1] = 0
        for i in train:
            resultsAux = knn2(data, i, k, 0, 0)
            results[0] += resultsAux[0]
            results[1] += resultsAux[1]
            resultsAux2[0] += resultsAux[0]
            resultsAux2[1] += resultsAux[1]
        accuracies2 += [resultsAux2[0] / (resultsAux2[1] + resultsAux2[0])]
        resultsAux2[0] = 0
        resultsAux2[1] = 0

    return [accuracies1, accuracies2]

def euclidianDistance(ponto1, ponto2):
    distance = 0
    for i in range(0, 9):
        distance += (ponto1[i] - ponto2[i]) ** 2
    distance = math.sqrt(distance)
    return distance

# EXERCISE 7
def splitData(list):
    a = []
    b = []
    for i in list:
        a.append(i[:-1])
        b.append(i[-1])
    return [a,b]

def naiveCalculate(train, test):
    train1 = []
    train2 = []
    accuracy = 0
    test1, test2 = splitData(test)
    train1, train2 = splitData(train)
    resAux = MultinomialNB()
    resAux.fit(train1, train2)
    previsions = resAux.predict(test1)
    for j in range(len(previsions)):
        if test2[j] == previsions[j]:
            accuracy = accuracy + 1
    finalAccuracy = accuracy/len(test)
    return finalAccuracy

def naive(data):
    result = []
    testData = []
    trainData = []
    for train, test in Res.split(data):
        for i in train:
            trainData += [data[i]]
        for i in test:
            testData += [data[i]]
        result += [naiveCalculate(trainData, testData)]
    return result

# MAIN FUNCTION

def main():
    res = []
    accuracy1 = 0
    accuracy2 = 0
    accuracy3 = 0
    #GET DATA
    k = eval(input("k: "))
    with open("HW1.txt") as f:
        lines = f.readlines()
    for line in lines:
        tmp = line.split(',')
        res.append(tmp)
    data = getDataToMatrix(res)
    # EXERCISE 5
    benign, malignant = divideData(data)
    fig, _ = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))  # creating the histograms
    fig.tight_layout(pad=4.0)
    axes = fig.axes #list with the axes
    fig.canvas.set_window_title('AP HW01 G132')  # Title of histogram
    colors = ['blue', 'red']

    # Titles of histograms
    for i in range(len(axes)):
        vectorsBenign = getVectors(benign, i)
        vectorsMalignant = getVectors(malignant, i)
        axes[i].title.set_text(atributes[i])
        axes[i].set_xlabel("Value")
        axes[i].legend(prop={'size': 9})
        axes[i].set_ylabel("Counter")
        axes[i].hist([vectorsBenign, vectorsMalignant], 10, density=False, histtype='bar', color=colors,
                    label=["Benign", "Malignant"])

    plt.show()

    # EXERCISE 6
    results1 = kFold1(data, k)
    for i in results1[0]:
        accuracy1 += i
    accuracy1 /= 10
    for i in results1[1]:
        accuracy2 += i
    accuracy2 /= 10
    print("accuracy test: " + str(accuracy1))
    print("accuracy train: " + str(accuracy2))

    # EXERCISE 7

    results3 = naive(data)
    for i in results3:
        accuracy3 += i
    accuracy3 /= 10
    print("accuracy bayes: " + str(accuracy3))

    pValue = stats.ttest_ind(np.array(results1[0]), np.array(results3), alternative="less")
    print(pValue)

main()