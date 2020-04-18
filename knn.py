import csv
import random

##
#
# load irist dataset and randomly split it into test set and training set
#
##

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

##
#
# euclidean distance calcualtion
#
##

import math
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)



###
#
# getting the nearest neighbors by selecting subset with the smallest distance
#
###

import operator 
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors



###
#
# Get the predicted group
#
###

import operator
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


###
#
# Measure the accuracy of the predictions
#
###
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] in predictions[x]: 
            correct = correct + 1
            
    return (correct/float(len(testSet))*100) 

def main():
    # prepare data
    trainingSet=[]
    testSet=[]

    # Defining the train and test split
    split = 0.70

    # load the dataset
    loadDataset('iris.csv', split, trainingSet, testSet)
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))

    # generate predictions
    predictions=[]
    k = 3
    print("Predicted\t\tActual")
    print("----------------------------------------")
    for x in range(len(testSet)):
        # Identify the k groups
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        # Get the predictions
        result = getResponse(neighbors)
        predictions.append(result)
        print('' + repr(result) + '\t\t' + repr(testSet[x][-1]))
    
    # Calculate the accuracy
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
if __name__ == "__main__":
    main()
