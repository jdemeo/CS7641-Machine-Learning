# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes
# to a file and plot them in your favorite tool.
import sys
import os
import time
import csv

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array




"""
Commandline parameter(s):
    none
"""

# set N value.  This is the number of points
N = 50
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

# ----------RUNNING MORE ITERATIONS FOR INSTANCE BASED ALGOS--------------

# Store Metrics
rhc_times = []
rhc_acc = []
sa_times = []
sa_acc = []
ga_times = []
ga_acc = []

NUMBER_ITERATIONS = 50000
for iteration in xrange(NUMBER_ITERATIONS):
    if iteration % 1000 == 0:
        rhc = RandomizedHillClimbing(hcp)
        fit = FixedIterationTrainer(rhc, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        rhc_times.append(end - start)
        rhc_acc.append(ef.value(rhc.getOptimal()))
        print "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))


        sa = SimulatedAnnealing(1E12, .999, hcp)
        fit = FixedIterationTrainer(sa, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        sa_times.append(end - start)
        sa_acc.append(ef.value(sa.getOptimal()))
        print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))

        ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
        fit = FixedIterationTrainer(ga, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        ga_times.append(end - start)
        ga_acc.append(ef.value(ga.getOptimal()))
        print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))
        print "---------------"
    else:
        continue

metrics = [rhc_acc, rhc_times, sa_acc, sa_times, ga_acc, ga_times]

# Write data to CSV file
with open("metrics/" + 'tsm_rhc_sa_ga_50000.csv', 'w') as f:
    writer = csv.writer(f)
    for metric in metrics:
        writer.writerow(metric)


# ------------RUNNING SAME NUMBER OF ITERATIONS ACROSS ALGORTIHMS--------

# Store Metrics
rhc_times = []
rhc_acc = []
sa_times = []
sa_acc = []
ga_times = []
ga_acc = []
mimic_times = []
mimic_acc = []

NUMBER_ITERATIONS = 1000
for iteration in xrange(NUMBER_ITERATIONS):
    if iteration % 10 == 0:
        rhc = RandomizedHillClimbing(hcp)
        fit = FixedIterationTrainer(rhc, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        rhc_times.append(end - start)
        rhc_acc.append(ef.value(rhc.getOptimal()))
        print "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))


        sa = SimulatedAnnealing(1E12, .999, hcp)
        fit = FixedIterationTrainer(sa, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        sa_times.append(end - start)
        sa_acc.append(ef.value(sa.getOptimal()))
        print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))

        ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
        fit = FixedIterationTrainer(ga, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        ga_times.append(end - start)
        ga_acc.append(ef.value(ga.getOptimal()))
        print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))

    else:
        continue

NUMBER_ITERATIONS = 1000
for iteration in xrange(NUMBER_ITERATIONS):
    if iteration % 10 == 0:
        # for mimic we use a sort encoding
        ef = TravelingSalesmanSortEvaluationFunction(points);
        fill = [N] * N
        ranges = array('i', fill)
        odd = DiscreteUniformDistribution(ranges);
        df = DiscreteDependencyTree(.1, ranges);
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

        mimic = MIMIC(500, 100, pop)
        fit = FixedIterationTrainer(mimic, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        mimic_times.append(end - start)
        mimic_acc.append(ef.value(mimic.getOptimal()))
        print "MIMIC Inverse of Distance: " + str(ef.value(mimic.getOptimal()))
        print "--------"

    else:
        continue

metrics = [rhc_acc, rhc_times, sa_acc, sa_times, ga_acc, ga_times, mimic_acc, mimic_times,]

# Write data to CSV file
with open("metrics/" + 'tsm_1000.csv', 'w') as f:
    writer = csv.writer(f)
    for metric in metrics:
        writer.writerow(metric)
