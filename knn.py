import operator
import math
import csv
import random

##
#
# Load the user provided dataset and 
# provide a random split it 
# Split into test set and training set
#
##

def load_data(fname, split, training_set=[] , test_set=[]):
    with open(fname, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset)-1):
            for y in range(len(dataset[x])-1):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])

##
#
# Euclidean dist calculation
# for identifying nearby nodes
#
##
def euclid_dist_calc(instance1, instance2, length):
    dist = 0
    for x in range(length):
        dist += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(dist)



###
#
# Getting the nearest neighbors by 
# selecting subset with the smallest 
# distance from node x
#
###
def get_near_neighbors(training_set, testInstance, k):
    dists = []
    length = len(testInstance)-1
    for x in range(len(training_set)):
        dist = euclid_dist_calc(testInstance, training_set[x], length)
        dists.append((training_set[x], dist))
    dists.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(dists[x][0])
    return neighbors



###
#
# Get the predicted group
#
###
def get_result_neighbour(neighbors):
    class_points = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_points:
            class_points[response] += 1
        else:
            class_points[response] = 1
    sortedVotes = sorted(class_points.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


###
#
# Measure the accuracy of the predictions
#
###
def get_stats(test_set, preds):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] in preds[x]: 
            correct = correct + 1
            
    return (correct/float(len(test_set))*100) 

def main():
    # prepare data
    training_set=[]
    test_set=[]

    # Defining the train and test split
    split = 0.70

    # load the dataset
    load_data('seeds.csv', split, training_set, test_set)
    print ('Train set size: ' + repr(len(training_set)))
    print ('Test set size: ' + repr(len(test_set)))

    # generate preds
    preds=[]
    k = 3
    print("Predicted\tActual")
    print("-----------\t------------")
    #print(training_set[0])
    #exit(0)
    for x in range(len(test_set)):
        # Identify the k groups
        neighbors = get_near_neighbors(training_set, test_set[x], k)
        # Get the preds
        result = get_result_neighbour(neighbors)
        preds.append(result)
        print('' + repr(result) + '\t\t' + repr(test_set[x][-1]))
    
    # Calculate the accuracy
    accuracy = get_stats(test_set, preds)
    print('Accuracy: ' + repr(accuracy) + '%')
    
if __name__ == "__main__":
    main()
