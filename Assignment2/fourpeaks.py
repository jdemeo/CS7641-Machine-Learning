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

from array import array



"""
Commandline parameter(s):
   none
"""


N=200
T=N/5
fill = [2] * N
ranges = array('i', fill)

ef = FourPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

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
        print "RHC: " + str(ef.value(rhc.getOptimal()))


        sa = SimulatedAnnealing(1E11, .95, hcp)
        fit = FixedIterationTrainer(sa, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        sa_times.append(end - start)
        sa_acc.append(ef.value(sa.getOptimal()))
        print "SA: " + str(ef.value(sa.getOptimal()))

        ga = StandardGeneticAlgorithm(200, 100, 10, gap)
        fit = FixedIterationTrainer(ga, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        ga_times.append(end - start)
        ga_acc.append(ef.value(ga.getOptimal()))
        print "GA: " + str(ef.value(ga.getOptimal()))

        mimic = MIMIC(200, 20, pop)
        fit = FixedIterationTrainer(mimic, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        mimic_times.append(end - start)
        mimic_acc.append(ef.value(mimic.getOptimal()))
        print "MIMIC: " + str(ef.value(mimic.getOptimal()))
        print "---------------"

    else:
        continue

metrics = [rhc_acc, rhc_times, sa_acc, sa_times, ga_acc, ga_times, mimic_acc, mimic_times,]

# Write data to CSV file
with open("metrics/" + 'fourpeaks_1000.csv', 'w') as f:
    writer = csv.writer(f)
    for metric in metrics:
        writer.writerow(metric)


# ----------RUNNING MORE ITERATIONS FOR RHC, SA, GA--------------

# Store Metrics
rhc_times = []
rhc_acc = []
sa_times = []
sa_acc = []
ga_times = []
ga_acc = []

NUMBER_ITERATIONS = 50000
for iteration in xrange(NUMBER_ITERATIONS):
    if iteration % 100 == 0:
        rhc = RandomizedHillClimbing(hcp)
        fit = FixedIterationTrainer(rhc, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        rhc_times.append(end - start)
        rhc_acc.append(ef.value(rhc.getOptimal()))
        print "RHC: " + str(ef.value(rhc.getOptimal()))


        sa = SimulatedAnnealing(1E11, .95, hcp)
        fit = FixedIterationTrainer(sa, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        sa_times.append(end - start)
        sa_acc.append(ef.value(sa.getOptimal()))
        print "SA: " + str(ef.value(sa.getOptimal()))

        ga = StandardGeneticAlgorithm(200, 100, 10, gap)
        fit = FixedIterationTrainer(ga, iteration)
        start = time.time()
        fit.train()
        end = time.time()
        ga_times.append(end - start)
        ga_acc.append(ef.value(ga.getOptimal()))
        print "GA: " + str(ef.value(ga.getOptimal()))
        print "---------------"

    else:
        continue

metrics = [rhc_acc, rhc_times, sa_acc, sa_times, ga_acc, ga_times]

# Write data to CSV file
with open("metrics/" + 'fourpeaks_rhc_sa_ga_50000.csv', 'w') as f:
    writer = csv.writer(f)
    for metric in metrics:
        writer.writerow(metric)
