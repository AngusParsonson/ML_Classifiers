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

# Feature Selection ------------------------------------------------------------------------------------------------------

def reduce_data(train_set, test_set, selected_features):
    train_set_red = train_set[:, selected_features]
    test_set_red  = test_set[:, selected_features]
    
    return train_set_red, test_set_red

def feature_selection(train_set, train_labels, **kwargs):

    number_of_features = 2
    train_set_red, test_set_red = reduce_data(train_set, test_set, [10,12])
    
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
    
def calculate_centroids(train_set, train_labels):
    classes = np.unique(train_labels)
    centroids = np.array([np.mean(train_set[train_labels == c, :], axis=0) for c in classes])
    
    return centroids, classes

def nearest_centroid(centroids, test_set):
    dist = lambda x,y : np.sqrt(np.sum((x-y)**2))
    centroid_dist = lambda x : [dist(x, centroid) for centroid in centroids]
    predicted = np.argmin([centroid_dist(p) for p in test_set], axis = 1).astype(np.int) + 1
    
    return predicted

def knn(train_set, train_labels, test_set, k, **kwargs):
    centroids, classes = calculate_centroids(train_set_red, train_labels)
    
    predicted = nearest_centroid(centroids, test_set_red)
    np.savetxt('results.csv', predicted, delimiter=',', fmt='%d')

    for i, pred in enumerate(predicted):
        print('{:02d} gt class: {}\tpredicted class: {}'.format(i+1, pred, test_labels[i]))

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


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
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
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