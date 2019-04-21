"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
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
        # optimal_instance = oa.getOptimal()
        # network.setWeights(optimal_instance.getData())

        test_time, accuracy = training_or_testing(test_data, network, start)
        test_times.append(test_time)
        test_acc.append(accuracy)

    metrics = [train_acc, train_times, test_acc, test_times]

    # Write data to CSV file
    with open("metrics/" + oaName + '_metrics.csv', 'w') as f:
        writer = csv.writer(f)
        for metric in metrics:
            writer.writerow(metric)


def main():
    # Optimization Algorithms
    # RandomizedHillClimbing(nnop)
    # SimulatedAnnealing(1E11, .95, nnop)  (TEMPERATURE, COOLING RATE)
    # StandardGeneticAlgorithm(200, 100, 10, nnop) (Population, ToMate, ToMutate)

    train_data = initialize_instances(TRAIN_FILE)
    test_data = initialize_instances(TEST_FILE)                 # Get data
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_data)
    activation = RELU()
    oa_name = 'SA'
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER], activation)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = SimulatedAnnealing(1E11, .80, nnop)
    train(oa, classification_network, oa_name, train_data, test_data, measure)

if __name__ == "__main__":
    main()
