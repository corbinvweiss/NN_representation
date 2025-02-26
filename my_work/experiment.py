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

NUM_EPOCHS = 10
LEARNING_RATE = 2.5

def interactive_SGD(mlp, training_data: np.array, epochs, mini_batch_size, eta,
                    classified_test_samples=None):
    """
    Train the neural network using mini-batch stochastic
    gradient descent. Pause after every epoch to write the
    predictions to the results.json file for all test data.
    This will allow us to analyze the changes in internal representation
    between training epochs.
    """
    n = len(training_data)
    epoch_classifications = []
    for j in range(epochs):
        # time1 = time.time()
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            mlp.update_mini_batch(mini_batch, eta)
        if(classified_test_samples):
            epoch_classifications.append(feedforward_samples(mlp, classified_test_samples))
            print(f"Completed epoch {j}.")

    if(classified_test_samples):
        return epoch_classifications

            
    # Copy the SGD code from network.py. At the end of each epoch run
    # the classified samples passed in classified_test_samples through the network. 
    # Write the results to results.json just as you do in main()
    # the next step will be to write a renderer to create a PCA viz for 
    # the network at each epoch and let the user navigate between them to watch it learn.

def interactive_training(training_data: np.array, classified_test_samples):
    """
    train a network using interactive Stochastic Gradient Descent,
    which will return a list of classifications of the classified_test_samples
    at the end of each epoch, reflecting the internal state of the MLP after each epoch.
    """
    random.seed(12345678)
    np.random.seed(12345678)
    net = network.Network([784, 30, 10])
    return interactive_SGD(net, training_data, NUM_EPOCHS, 10, LEARNING_RATE, classified_test_samples)


def train_network(training_data: np.array) -> network.Network:
    """
    Train the network based on the mnist dataset
    """
    random.seed(12345678)
    np.random.seed(12345678)
    net = network.Network([784, 30, 10])
    net.SGD(training_data, NUM_EPOCHS, 10, LEARNING_RATE, test_data=None)
    return net

def classify_samples(samples, count=None):
    """
    split the test data samples into the 10 classes.
    return a list of lists of images, where the index is the class of the image
    optional parameter count lets you specify how many of each class to return
    """
    classified = [[] for i in range(10)]
    for sample in samples:
        if(not (count and len(classified[sample[1]]) >= count) ):
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

def write_results(results, filename):
    with open(filename, mode="w", encoding="utf-8") as dest:
        json.dump(results, dest)

def fully_trained_MLP():
    """
    prepare data representing state of fully trained MLP for use with the fullyTrained.ipynb notebook
    Writes a list of classifications for each class of handwritten digit to results.json
    """
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 
    classified_samples = classify_samples(test_data)
    MLP = train_network(training_data)
    classifications = feedforward_samples(MLP, classified_samples)
    write_results(classifications, "fullyTrained.json")


def MLP_in_training():
    """
    prepare data representing the state of an MLP in training for use with inTraining.ipynb notebook
    Writes a list of classifications for each class of handwritten digits for each epoch of training
    """
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 
    classified_samples = classify_samples(test_data, 500)
    epoch_classifications = interactive_training(training_data, classified_test_samples=classified_samples)
    write_results(epoch_classifications, "inTraining.json")

if __name__ == "__main__":
    MLP_in_training()
