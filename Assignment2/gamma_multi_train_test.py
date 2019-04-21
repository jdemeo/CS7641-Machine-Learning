"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network for the GAMMA dataset

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU
from helper_functions import training_or_testing

TRAIN_FILE = os.path.join("..", "datasets", "gamma_train.txt")
TEST_FILE = os.path.join("..", "datasets", "gamma_test.txt")

INPUT_LAYER = 10
HIDDEN_LAYER1 = 10
HIDDEN_LAYER2 = 10
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5000


def initialize_instances(examples):
    """Read CSV data into a list of instances."""
    instances = []

    # Read in the abalone.txt CSV file
    with open(examples, "r") as gamma:
        reader = csv.reader(gamma)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if row[-1] == 'g' else 1))
            instances.append(instance)

    return instances


def train(oa, network, oaName, train_data, test_data, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print "\nError results for %s\n---------------------------" % (oaName,)

    # Storing times and accuracies overtime
    train_times = []
    train_acc = []
    test_times = []
    test_acc = []

    for iteration in xrange(TRAINING_ITERATIONS):
        '''TRAINING'''
        start = time.time()
        oa.train()

        train_time, accuracy = training_or_testing(train_data, network, start)
        train_times.append(train_time)
        train_acc.append(accuracy)

        '''TESTING'''
        # Get optimal network architecture from training
        start = time.time()
        optimal_instance = oa.getOptimal()
        network.setWeights(optimal_instance.getData())

        test_time, accuracy = training_or_testing(test_data, network, start)
        test_times.append(test_time)
        test_acc.append(accuracy)

    metrics = [train_acc, train_times, test_acc, test_times]

    return metrics

def main():
    """
    Run algorithms on the gamma dataset.
    Essentially ran twice for 2-fold cross validation
    Metrics are evaluated outside of this file
    """
    train_data = initialize_instances(TRAIN_FILE)
    test_data = initialize_instances(TEST_FILE)                 # Get data
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_data)

    networks = []  # BackPropagationNetwork
    nnop = []      # NeuralNetworkOptimizationProblem
    oa = []        # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA"]
    results = ""

    # Create each network architecture and an optimization instance
    for name in oa_names:
        activation = RELU()
        # Change network size
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER], activation)
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    # Randomized Optimzation Algos
    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
    oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    # Go through each optimization problem and do 2-fold CV
    for i, name in enumerate(oa_names):
        start = time.time()
        metrics = train(oa[i], networks[i], oa_names[i], train_data, test_data, measure)
        end = time.time()
        training_time = end - start
        results += "\nFold 1 train time: %0.03f seconds" % (training_time,)

        # Write data to CSV file
        with open("metrics/" + oa_names[i] + '_f1.csv', 'w') as f:
            writer = csv.writer(f)
            for metric in metrics:
                writer.writerow(metric)

    print results

    # 2nd fold;
    train_data = initialize_instances(TEST_FILE)
    test_data = initialize_instances(TRAIN_FILE)                 # Get data
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_data)

    networks = []  # BackPropagationNetwork
    nnop = []      # NeuralNetworkOptimizationProblem
    oa = []        # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA"]
    results = ""

    # Create each network architecture and an optimization instance
    for name in oa_names:
        activation = RELU()
        # Change network size
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER], activation)
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    # Randomized Optimzation Algos
    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
    oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    # Go through each optimization problem and do 2-fold CV
    for i, name in enumerate(oa_names):
        start = time.time()
        metrics = train(oa[i], networks[i], oa_names[i], train_data, test_data, measure)
        end = time.time()
        training_time = end - start
        results += "\nFold 1 train time: %0.03f seconds" % (training_time,)

        # Write data to CSV file
        with open("metrics/" + oa_names[i] + '_f2.csv', 'w') as f:
            writer = csv.writer(f)
            for metric in metrics:
                writer.writerow(metric)

    print results


if __name__ == "__main__":
    main()
