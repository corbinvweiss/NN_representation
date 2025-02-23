# experiment.py
# Corbin Weiss
# 2025-2-21
# code for the neural network representation experiment

import sys
import json
import random
import numpy as np

sys.path.append('../src/')
import network
import mnist_loader

NUM_EPOCHS = 5
LEARNING_RATE = 2.5

def train_network(training_data: np.array) -> network.Network:
    """
    Train the network based on the mnist dataset
    """
    random.seed(12345678)
    np.random.seed(12345678)
    net = network.Network([784, 30, 10])
    net.SGD(training_data, NUM_EPOCHS, 10, LEARNING_RATE, test_data=None)
    return net

def classify_samples(samples):
    """
    split the test data samples into the 10 classes.
    return a list of lists of images, where the index is the class of the image
    """
    classified = [[] for i in range(10)]
    for sample in samples:
        classified[sample[1]].append(sample[0])

    return classified


def feedforward_samples(mlp, classified_samples):
    """
    run a set of samples through the mlp and see how it classifies them
    classified_samples: [[images of 0], [images of 1], ... [images of 9]]
    returns a list of classifications [[classifications of 0], [classifications of 1], ..., [classifications of 9]]
    """
    classifications = [[] for i in range(10)]
    for i in range(10):
        for sample in classified_samples[i]:
            classifications[i].append([val[0] for val in mlp.feedforward(sample)])

    return classifications
        
def mean_predictions(mlp_classifications):
    """
    average the value of each value in the prediction vector across all samples of each class
    Average across predictions such as 
    [[[0, 0, 0.01, 0.98, 0, 0, 0, 0.05, 0, 0]],
    [[0, 0, 0.01, 0.98, 0, 0, 0.3, 0.05, 0, 0],
    [0, 0, 0.01, 0.98, 0.2, 0, 0, 0.05, 0, 0]]... ]
    for all samples of each class
    the outermost index is the class of those sample predictions
    the next one in is the predictions themselves
    inside that is the prediction index
    we want to add up the predictions by index across the predictions of each class.
    """
    mean_predictions = [[] for i in range(10)]  # mean values of each prediction index for each class
    for class_label in range(10):
        class_mean_predictions = [[] for i in range(10)]    # mean values of each prediction index for that class
        for pred_index in range(10):
            index_predictions = []    # actual prediction values of that index for samples of the class
            for sample in mlp_classifications[class_label]:
                index_predictions.append(sample[pred_index])
            class_mean_predictions[pred_index].append(np.mean(index_predictions))
        mean_predictions[class_label].append(class_mean_predictions)
        
    return mean_predictions

def write_results(results):
    with open("results.json", mode="w", encoding="utf-8") as dest:
        json.dump(results, dest)

def main():
    """
    get the training data, train the network.
    split the testing data into its classes.
    Run the the classified testing data through the trained network and record 
    the classification vectors
    """
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 
    classified_samples = classify_samples(test_data)
    MLP = train_network(training_data)

    mlp_classifications = feedforward_samples(MLP, classified_samples)
    print(mlp_classifications)
    write_results(mlp_classifications)
    print(mean_predictions(mlp_classifications))

if __name__ == "__main__":
    main()
