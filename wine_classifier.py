#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission. 
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']    

# Accuracy calculations --------------------------------------------------------------------------------------------------

def calculate_accuracy(gt_labels, pred_labels):
    total_correct_predictions = sum(gt_labels == pred_labels)   
    accuracy = (total_correct_predictions/len(gt_labels))*100
    
    return accuracy

def calculate_confusion_matrix(gt_labels, pred_labels, num_classes):
    # write your code here (remember to return the confusion matrix at the end!)   
    matrix = np.zeros([num_classes, num_classes])
    for i in range (num_classes):
        for j in range (num_classes):
            incorrectly_predicted = 0
            number_of_i = 0
            
            for x in range (len(gt_labels)):
                if (gt_labels[x] == i+1 and pred_labels[x] == j+1):
                    incorrectly_predicted += 1
                    number_of_i += 1
                    
                elif (gt_labels[x] == i+1):
                    number_of_i += 1
           
            if (number_of_i != 0 ):  
                matrix[i][j] = incorrectly_predicted/number_of_i
            else:
                matrix[i][j] = 0
    
    return matrix

# Feature Selection ------------------------------------------------------------------------------------------------------

def reduce_data(train_set, test_set, selected_features):
    
    train_set_red = train_set[:, selected_features]
    test_set_red  = test_set[:, selected_features]
    
    return train_set_red, test_set_red

def feature_selection(train_set, train_labels, **kwargs):
    
    number_of_features = 2
    train_set_red, test_set_red = reduce_data(train_set, test_set, [0,11])
    
    n_features = train_set.shape[1]
    fig, ax = plt.subplots(number_of_features, number_of_features)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

    class_1_colour = r'#3366ff'
    class_2_colour = r'#cc3300'
    class_3_colour = r'#ffc34d'

    class_colours = [class_1_colour, class_2_colour, class_3_colour]

    # write your code here
    colours = np.zeros_like(train_labels, dtype=np.object)
    colours[train_labels == 1] = class_1_colour
    colours[train_labels == 2] = class_2_colour
    colours[train_labels == 3] = class_3_colour

    for row in range(number_of_features):
        for col in range(number_of_features):
            ax[row][col].scatter(train_set_red[:, row], train_set_red[:, col], c=colours)
            ax[row][col].set_title('Features {} vs {}'.format(row+1, col+1))

    plt.show()
    
# Knn -------------------------------------------------------------------------------------------------------------------

def knn(train_set, train_labels, test_set, k, **kwargs):
    
    # Only uses the chosen two features to train and test the classifier
    number_of_features = 2
    train_set_red, test_set_red = reduce_data(train_set, test_set, [0,11])
    
    number_of_test_items  = len(test_set_red)
    number_of_train_items = len(train_set_red)
    
    dist = lambda x,y : np.sqrt(np.sum((x-y)**2))
    distances_to_points = np.zeros(shape=(number_of_test_items,number_of_train_items))

    for i in range (number_of_test_items):
        for j in range (number_of_train_items):
            distances_to_points[i][j] = dist(test_set_red[i], train_set_red[j])
        
    # Obtains an array of size test_set, with each item containing an array of the indices of the k closest training items
    k_nearest_neighbours = np.zeros(shape=(number_of_test_items,k))
    knn_classes = np.zeros(shape=(number_of_test_items,k), dtype=int)
    
    for i in range (number_of_test_items):
        k_nearest_neighbours[i] = np.argsort(distances_to_points[i])[:k]
        
    ''' #Prints the distances of nearest points to test items
    for i in range (number_of_test_items):
        print( )
        for j in range (k):
            print(distances_to_points[i][k_nearest_neighbours[i][j].astype(np.int)])
    '''
    for i in range (number_of_test_items):
        for j in range (k):
            knn_classes[i][j] = train_labels[k_nearest_neighbours[i][j].astype(np.int)]
    
    predictions = np.zeros(number_of_test_items, dtype=int)
    
    for i in range (number_of_test_items):
        predictions[i] = np.bincount(knn_classes[i]).argmax()
        
    print(calculate_accuracy(test_labels, predictions))
    matrix = calculate_confusion_matrix(test_labels, predictions, 3)
    #print(matrix)
    print(predictions)
    
    return predictions

# Decision tree -----------------------------------------------------------------------------------------------------------
# Tree datastructure copied from stack overflow and modified https://stackoverflow.com/questions/2598437/how-to-implement-a-binary-tree   
        
def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    
    # Only uses the chosen two features to train and test the classifier
    number_of_features = 2
    train_set_red, test_set_red = reduce_data(train_set, test_set, [0,11])
    
    number_of_test_items  = len(test_set_red)
    number_of_train_items = len(train_set_red)
    
    classes = [1,2,3]
    means = np.array([np.mean(train_set_red[train_labels == c, :], axis=0) for c in classes])
    standard_deviations = np.array([np.std(train_set_red[train_labels == c, :], axis=0) for c in classes])   
    
    class_likelihoods = np.bincount(train_labels - 1) / len(train_labels)
     
    predicted_class = np.zeros(shape=(number_of_test_items), dtype=int)
    for i in range (number_of_test_items):
        class_probabilities = np.zeros(shape=(3))
        
        for j in range (3):
            x = 1 / np.sqrt(2 * np.pi * standard_deviations[j][0])
            y = - ((test_set_red[i][0] - means[j][0])**2) / (2 * standard_deviations[j][0])
            probability_first_attribute = x * np.exp(y)
            
            x = 1 / np.sqrt(2 * np.pi * standard_deviations[j][1])
            y = - ((test_set_red[i][1] - means[j][1])**2) / (2 * standard_deviations[j][1])
            probability_second_attribute = x * np.exp(y)
            
            class_probabilities[j] = (probability_first_attribute * probability_second_attribute) * class_likelihoods[j]
        
        predicted_class[i] = class_probabilities.argmax() + 1
            
    print(calculate_accuracy(test_labels, predicted_class))
    return predicted_class


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path, 
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        knn(train_set, train_labels, test_set, args.k)
        #predictions = knn(train_set, train_labels, test_set, args.k)
        #print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))