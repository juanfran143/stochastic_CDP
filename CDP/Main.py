import sys
from Instance import Instance
from objects import Test
import random
import os
from ConstructiveHeuristic import ConstructiveHeuristic
import time
from LocalSearches import tabuSearch, tabuSearch_capacity_simulation
from LocalSearches import tabuSearch_capacity
import numpy as np
from simheuristic import *


def build_run_file():
    fileName = 'test/test2run.txt'

    instances = ["GKD-b_11_n50_b02_m5.txt", "GKD-b_12_n50_b02_m5.txt", "GKD-b_13_n50_b02_m5.txt", "GKD-b_14_n50_b02_m5.txt",
    "GKD-b_15_n50_b02_m5.txt", "GKD-b_16_n50_b02_m15.txt", "GKD-b_17_n50_b02_m15.txt", "GKD-b_18_n50_b02_m15.txt", "GKD-b_19_n50_b02_m15.txt",
    "GKD-b_20_n50_b02_m15.txt", "GKD-b_41_n150_b02_m15.txt", "GKD-b_42_n150_b02_m15.txt", "GKD-b_43_n150_b02_m15.txt", "GKD-b_44_n150_b02_m15.txt",
    "GKD-b_45_n150_b02_m15.txt", "GKD-b_46_n150_b02_m45.txt", "GKD-b_47_n150_b02_m45.txt", "GKD-b_48_n150_b02_m45.txt",
    "GKD-b_49_n150_b02_m45.txt", "GKD-b_50_n150_b02_m45.txt", "GKD-b_11_n50_b03_m5.txt", "GKD-b_12_n50_b03_m5.txt", "GKD-b_13_n50_b03_m5.txt", "GKD-b_14_n50_b03_m5.txt",
    "GKD-b_15_n50_b03_m5.txt", "GKD-b_16_n50_b03_m15.txt", "GKD-b_17_n50_b03_m15.txt", "GKD-b_18_n50_b03_m15.txt", "GKD-b_19_n50_b03_m15.txt",
    "GKD-b_20_n50_b03_m15.txt", "GKD-b_41_n150_b03_m15.txt", "GKD-b_42_n150_b03_m15.txt", "GKD-b_43_n150_b03_m15.txt", "GKD-b_44_n150_b03_m15.txt",
    "GKD-b_45_n150_b03_m15.txt", "GKD-b_46_n150_b03_m45.txt", "GKD-b_47_n150_b03_m45.txt", "GKD-b_48_n150_b03_m45.txt",
    "GKD-b_49_n150_b03_m45.txt", "GKD-b_50_n150_b03_m45.txt"]

    instances = ["GKD-b_11_n50_b03_m5.txt", "GKD-b_12_n50_b03_m5.txt", "GKD-b_13_n50_b03_m5.txt", "GKD-b_14_n50_b03_m5.txt",
    "GKD-b_15_n50_b03_m5.txt", "GKD-b_16_n50_b03_m15.txt", "GKD-b_17_n50_b03_m15.txt", "GKD-b_18_n50_b03_m15.txt", "GKD-b_19_n50_b03_m15.txt",
    "GKD-b_20_n50_b03_m15.txt", "GKD-b_41_n150_b03_m15.txt", "GKD-b_42_n150_b03_m15.txt", "GKD-b_43_n150_b03_m15.txt", "GKD-b_44_n150_b03_m15.txt",
    "GKD-b_45_n150_b03_m15.txt", "GKD-b_46_n150_b03_m45.txt", "GKD-b_47_n150_b03_m45.txt", "GKD-b_48_n150_b03_m45.txt",
    "GKD-b_49_n150_b03_m45.txt", "GKD-b_50_n150_b03_m45.txt", ]

    """
    "SOM-a_11_n50_b02_m5.txt", "SOM-a_12_n50_b02_m5.txt",
    "SOM-a_13_n50_b02_m5.txt", "SOM-a_14_n50_b02_m5.txt", "SOM-a_15_n50_b02_m5.txt", "SOM-a_16_n50_b02_m15.txt",
    "SOM-a_17_n50_b02_m15.txt", "SOM-a_18_n50_b02_m15.txt", "SOM-a_19_n50_b02_m15.txt", "SOM-a_20_n50_b02_m15.txt"]
    
"""
    """
    instances = ["MDG-b_01_n500_b02_m50.txt", "MDG-b_02_n500_b02_m50.txt", "MDG-b_03_n500_b02_m50.txt", "MDG-b_04_n500_b02_m50.txt",
                 "MDG-b_05_n500_b02_m50.txt", "MDG-b_06_n500_b02_m50.txt", "MDG-b_07_n500_b02_m50.txt", "MDG-b_08_n500_b02_m50.txt"
                 , "MDG-b_09_n500_b02_m50.txt", "MDG-b_10_n500_b02_m50.txt",
                 "GKD-c_01_n500_b02_m50.txt", "GKD-c_02_n500_b02_m50.txt", "GKD-c_03_n500_b02_m50.txt", "GKD-c_04_n500_b02_m50.txt"
                 , "GKD-c_05_n500_b02_m50.txt", "GKD-c_06_n500_b02_m50.txt", "GKD-c_07_n500_b02_m50.txt", "GKD-c_08_n500_b02_m50.txt"
                 , "GKD-c_09_n500_b02_m50.txt", "GKD-c_10_n500_b02_m50.txt"]

    
    instances = ["MDG-b_01_n500_b03_m50.txt", "MDG-b_02_n500_b03_m50.txt", "MDG-b_03_n500_b03_m50.txt", "MDG-b_04_n500_b03_m50.txt",
                 "MDG-b_05_n500_b03_m50.txt", "MDG-b_06_n500_b03_m50.txt", "MDG-b_07_n500_b03_m50.txt", "MDG-b_08_n500_b03_m50.txt"
                 , "MDG-b_09_n500_b03_m50.txt", "MDG-b_10_n500_b03_m50.txt",
                 "GKD-c_01_n500_b03_m50.txt", "GKD-c_02_n500_b03_m50.txt", "GKD-c_03_n500_b03_m50.txt", "GKD-c_04_n500_b03_m50.txt"
                 , "GKD-c_05_n500_b03_m50.txt", "GKD-c_06_n500_b03_m50.txt", "GKD-c_07_n500_b03_m50.txt", "GKD-c_08_n500_b03_m50.txt"
                 , "GKD-c_09_n500_b03_m50.txt", "GKD-c_10_n500_b03_m50.txt"]
    
                 "GKD-b_50_n150_b02_m45.txt", "GKD-c_01_n500_b02_m50.txt", "GKD-c_02_n500_b02_m50.txt",
    "GKD-c_03_n500_b02_m50.txt", "GKD-c_04_n500_b02_m50.txt", "GKD-c_05_n500_b02_m50.txt", "GKD-c_06_n500_b02_m50.txt",
    "GKD-c_07_n500_b02_m50.txt", "GKD-c_08_n500_b02_m50.txt", "GKD-c_09_n500_b02_m50.txt", "GKD-c_10_n500_b02_m50.txt",
    
    instances = ["MDG-b_01_n500_b02_m50.txt", "MDG-b_02_n500_b02_m50.txt", "MDG-b_03_n500_b02_m50.txt",
                 "MDG-b_04_n500_b02_m50.txt",
                 "MDG-b_05_n500_b02_m50.txt", "MDG-b_06_n500_b02_m50.txt", "MDG-b_07_n500_b02_m50.txt",
                 "MDG-b_08_n500_b02_m50.txt"
        , "MDG-b_09_n500_b02_m50.txt", "MDG-b_10_n500_b02_m50.txt",
                                       "GKD-c_01_n500_b02_m50.txt", "GKD-c_02_n500_b02_m50.txt",
                 "GKD-c_03_n500_b02_m50.txt", "GKD-c_04_n500_b02_m50.txt"
        , "GKD-c_05_n500_b02_m50.txt", "GKD-c_06_n500_b02_m50.txt", "GKD-c_07_n500_b02_m50.txt",
                 "GKD-c_08_n500_b02_m50.txt"
        , "GKD-c_09_n500_b02_m50.txt", "GKD-c_10_n500_b02_m50.txt"]
"""
    #seeds = [23456, 764534, 6787654, 111134, 583483, 398582,843732,9922523,    5161240,5768318,6375397,6982475,7589554,8196632,8803711,9410789,10017868,10624946,11232025,11839103,]
    seeds = [23456, 764534] #   , 5161240, 5768318, 6375397 5161240,5768318,6375397,6982475,7589554,8196632,8803711,9410789,10017868,10624946,11232025,11839103]
    #seeds = [6787654, 111134, 583483, 398582, 843732]
    weight = [0.8]
    inversas = [0]
    time = 60
    beta = 0.3
    betaLs = 0.9999
    MaxIterLS = 50
    delta = 0.9
    short_simulation = 100
    long_simulation = 1000
    #vars = [0.1, 0.2, 0.25]
    vars = [0.1, 0.15, 0.2]
    deterministics = [False]
    not_penalization_costs = [False]#[True, False]

    for inversa in inversas:
        for w in weight:
            if not os.path.exists(fileName):
                with open(fileName, "w") as out:
                    #Instance   Seed	Time	Beta	BetaLs  MaxIterLS   delta   short_simulation	long_simulation   var   deterministic	not_penalization_cost   weight
                    out.write(
                        "#Instance\t" + "Seed\t" + "Time\t" + "Beta\t" + "BetaLs\t" + "MaxIterLS\t" + "delta\t" + "short_simulation\t" + "long_simulation\t" + "var\t" + "deterministic\t" + "not_penalization_cost\t" + "weight\t" + "inversa" +"\n")
            with open(fileName, "a") as out:
                for inst in instances:
                    for var in vars:
                        for seed in seeds:
                            for deterministic in deterministics:
                                if not deterministic:
                                    for not_penalization_cost in not_penalization_costs:
                                        out.write(str(inst) + "\t"+str(seed)+"\t"+str(time)+"\t"+str(beta)+"\t"+str(betaLs)+"\t"+str(MaxIterLS)+"\t"+str(delta)+"\t"+str(short_simulation)+"\t"+str(long_simulation)+"\t"+str(var)+"\t"+str(deterministic)+"\t"+str(not_penalization_cost)+"\t"+str(w)+ "\t" + str(inversa) +"\n")
                                if deterministic:
                                    out.write(str(inst) + "\t" + str(seed) + "\t" + str(time) + "\t" + str(beta) + "\t" + str(
                                        betaLs) + "\t" + str(MaxIterLS) + "\t" + str(delta) + "\t" + str(
                                        short_simulation) + "\t" + str(long_simulation) + "\t" + str(var) + "\t" + str(
                                        deterministic) + "\t" + "-" + "\t" + str(w) + "\t" + str(inversa) + "\n")
'''
Function to read de the testFile
The testife is composed of the following parameters:
#Instance   Seed    Time    BetaBR    BetaLs  MaxIterLS
-Instance: Name the instance
-Seed: Seed used to generate random numbers in the BR heuristic
-Time: Maximum execution time
-BetaBR: Beta parameter used in the BR heuristic
-BetaLs: Beta parameter used in the Local search
Note: Use # to comment lines in the file
'''
def readTest(testName):
    fileName = 'test' + os.sep + testName + '.txt'
    tests = []
    with open(fileName, 'r') as testsfile:
        for linetest in testsfile:
            linetest = linetest.strip()
            if '#' not in linetest:
                line = linetest.split('\t')
                test = Test(*line)
                tests.append(test)
    return tests


'''
This function creates a file with a summary of the executed instances. The
output will be: InstanceName BetaBR cost and time
where:
- InstanceName: Instance Name
- BetaBR: Beta used  in the BR Heuristic to obtain the solution 
- cost: Objective Function (Distance) 
- time: Time in which the solution has been obtained
'''
def writeData(sol, test):
        fileName = 'output' + os.sep + "ResumeOutputs_paper_2.txt"

        if not os.path.exists(fileName):
            with open(fileName, "w") as out:
                out.write(
                    "Instance\t" + "betaLS\t" + "CostSol\t" + "time\t" + "Capacity\t" + "reliability\t" + "variance\t" + "stochastic_of\t" + "stochastic_capacity\t" + "deterministic\t" + "type_simulation\t" + "inversa\t" + "weight\t"+ "seed\n")
        with open(fileName, "a") as out:
            if t.deterministic:
                out.write(test.instName + "\t" + str(test.betaBR) + "\t" + str(sol.of) + "\t" + str(sol.time) + "\t" + str(sol.capacity) + "\t" + str(sol.reliability["1"]) + "\t" + str(test.var) + "\t" + str(sol.of) + "\t" + str(sol.total_stochastic_capacity["1"]) + "\t" + str(t.deterministic) + "\t" + "True"+ "\t" + str(t.inversa) + "\t" + str(test.weight)+ "\t" + str(test.seed) + "\n")
                out.write(
                    test.instName + "\t" + str(test.betaBR) + "\t" + str(sol.of) + "\t" + str(sol.time) + "\t" + str(
                        sol.capacity) + "\t" + str(sol.reliability["2"]) + "\t" + str(test.var) + "\t" + str(
                        np.mean(sol.stochastic_of["2"])) + "\t" + str(sol.total_stochastic_capacity["2"]) + "\t" + str(test.deterministic) + "\t" + "False" + "\t" + str(t.inversa) + "\t" + str(test.weight)+ "\t" + str(test.seed) + "\n")
            else:
                if t.not_penalization_cost:
                    out.write(test.instName + "\t" + str(test.betaBR) + "\t" + str(sol.of) + "\t" + str(
                        sol.time) + "\t" + str(sol.capacity) + "\t" + str(sol.reliability["1"]) + "\t" + str(
                        test.var) + "\t" + str(np.mean(sol.stochastic_of["2"])) + "\t" + str(sol.total_stochastic_capacity["1"]) + "\t" + str(
                        t.deterministic) + "\t" + str(test.not_penalization_cost)+ "\t" + str(test.inversa)  + "\t" +  str(test.weight) + "\t" + str(test.seed) + "\n")
                else:
                    out.write(
                        test.instName + "\t" + str(test.betaBR) + "\t" + str(sol.of) + "\t" + str(
                            sol.time) + "\t" + str(
                            sol.capacity) + "\t" + str(sol.reliability["2"]) + "\t" + str(test.var) + "\t" + str(
                            np.mean(sol.stochastic_of["2"])) + "\t" + str(
                            sol.total_stochastic_capacity["2"]) + "\t" + str(test.deterministic) + "\t" + str(
                            test.not_penalization_cost)+ "\t" + str(test.inversa) + "\t" + str(test.weight)+ "\t" + str(test.seed) + "\n")


def writeData_det(sol, test):
    fileName = 'output' + os.sep + "ResumeOutputs_def_STOCHASTIC.txt"

    if not os.path.exists(fileName):
        with open(fileName, "w") as out:
            out.write(
                "Instance\t" + "betaLS\t" + "seed\t" + "CostSol\t" + "time\t" + "Capacity\t" + "inversa\t" + "weight\n")
    with open(fileName, "a") as out:
        out.write(test.instName + "\t" + str(test.betaBR) + "\t" + str(test.seed) + "\t" + str(sol.of) + "\t" + str(sol.time) + "\t" + str(
                sol.capacity) +"\t" + str(t.inversa) + "\t" + str(test.weight) + "\n")


def stochastic_multistart(bestSol, t: Instance, cl):
    var = t.var
    small_simulation = simheuristic(t.short_simulation, var)

    if t.not_penalization_cost:
        small_simulation.simulation_1(bestSol)
        small_simulation.simulation_2(bestSol, cl)
    else:
        small_simulation.simulation_2(bestSol, cl)

    #print("Initial Solution:", bestSol.of)

    elapsed = 0.0
    iter = 0
    elite_simulations = []
    elite_enter_simulations = []
    elite_simulations.append(bestSol)
    start = time.process_time()
    bestSol_axu = bestSol.copySol()

    enter = False
    while elapsed < t.Maxtime:
        iter += 1
        newSol, cl = heur.constructBRSol_capacity()  # biased-randomized version of the heuristic
        newSol = tabuSearch_capacity(newSol, cl, t.maxIter, heur)  # Local Search (Tabu Search)
        if newSol.of > bestSol.of:  # Check if the new solution improves the BestSol
            if t.not_penalization_cost:
                small_simulation.simulation_1(newSol)
                #small_simulation.simulation_2(newSol)
                if newSol.reliability["1"] >= t.delta:
                    enter = True
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)
                    #print("New Best Time:", elapsed)
                    #print("New Best Solution:", bestSol.of)
                elif not enter and newSol.reliability["1"] >= bestSol_axu.reliability["1"]:
                    bestSol_axu = newSol.copySol()  # Update Solution
                    bestSol_axu.time = elapsed
                    elite_enter_simulations.append(bestSol_axu)
            else:
                small_simulation.simulation_2(newSol, cl)
                #print("best: "+ str(bestSol.mean_stochastic_of["2"]) + " New_sol: " + str(newSol.mean_stochastic_of["2"])+ " tiempo:"+str(elapsed))
                if newSol.mean_stochastic_of["2"] >= bestSol.mean_stochastic_of["2"]:
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)
                    #print("New Best Time:", elapsed)
                    #print("New Best Solution:", bestSol.of)

        elapsed = time.process_time() - start
    #print(iter)
    large_simulation = simheuristic(t.long_simulation, var)
    if not enter and t.not_penalization_cost and len(elite_enter_simulations)>0:
        for i in elite_enter_simulations:
            large_simulation.simulation_1(i)
        elite_enter_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
        return elite_enter_simulations[0]

    for i in elite_simulations:
        if t.not_penalization_cost:
            large_simulation.simulation_1(i)
            #large_simulation.simulation_2(i)
        else:
            large_simulation.simulation_2(i, cl)

    if t.not_penalization_cost:
        elite_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
    else:
        elite_simulations.sort(key=lambda x: x.stochastic_of["2"], reverse=True)

    return elite_simulations[0]


def stochastic_multistart_simulation(bestSol, t: Instance):
    var = t.var
    small_simulation = simheuristic(t.short_simulation, var)

    bestSol, cl = heur.constructBRSol_capacity_simulation(simheuristic(20, var),
                                                         0.9)  # biased-randomized version of the heuristic
    bestSol = tabuSearch_capacity_simulation(bestSol, cl, t.maxIter, heur, simheuristic(20, var),
                                            0.9)  # Local Search (Tabu Search)

    if t.not_penalization_cost:
        small_simulation.simulation_1(bestSol)
    else:
        small_simulation.simulation_2(bestSol, cl)

    #print("Initial Solution:", bestSol.of)

    elapsed = 0.0
    iter = 0
    elite_simulations = []
    elite_enter_simulations = []
    elite_simulations.append(bestSol)
    start = time.process_time()
    bestSol_axu = bestSol.copySol()

    enter = False
    while elapsed < t.Maxtime:
        iter += 1
        newSol, cl = heur.constructBRSol_capacity_simulation(simheuristic(20, var), 0.9)  # biased-randomized version of the heuristic
        newSol = tabuSearch_capacity_simulation(newSol, cl, t.maxIter, heur, simheuristic(20, var), 0.9)  # Local Search (Tabu Search)
        if newSol.of > bestSol.of:  # Check if the new solution improves the BestSol
            if t.not_penalization_cost:
                small_simulation.simulation_1(newSol)
                #print(newSol.reliability["1"])
                if newSol.reliability["1"] >= t.delta:
                    enter = True
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)
                    #print("New Best Time:", elapsed)
                    #print("New Best Solution:", bestSol.of)
                elif not enter and newSol.reliability["1"] >= bestSol_axu.reliability["1"]:
                    bestSol_axu = newSol.copySol()  # Update Solution
                    bestSol_axu.time = elapsed
                    elite_enter_simulations.append(bestSol_axu)
            else:
                small_simulation.simulation_2(newSol, cl)
                #print("best: "+ str(bestSol.mean_stochastic_of["2"]) + " New_sol: " + str(newSol.mean_stochastic_of["2"])+ " tiempo:"+str(elapsed))
                if newSol.mean_stochastic_of["2"] >= bestSol.mean_stochastic_of["2"]:
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)
                    #print("New Best Time:", elapsed)
                    #print("New Best Solution:", bestSol.of)

        elapsed = time.process_time() - start
    #print(iter)
    large_simulation = simheuristic(t.long_simulation, var)
    if not enter and t.not_penalization_cost and len(elite_enter_simulations)>0:
        for i in elite_enter_simulations:
            large_simulation.simulation_1(i)
        elite_enter_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
        return elite_enter_simulations[0]

    for i in elite_simulations:
        if t.not_penalization_cost:
            large_simulation.simulation_1(i)
            #large_simulation.simulation_2(i, cl)
        else:
            large_simulation.simulation_2(i, cl)

    if t.not_penalization_cost:
        elite_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
        #elite_simulations.sort(key=lambda x: x.stochastic_of["2"], reverse=True)
    else:
        elite_simulations.sort(key=lambda x: x.stochastic_of["2"], reverse=True)

    return elite_simulations[0]

def deterministic_multistart(bestSol: Solution, t: Instance)->Solution:
    var = t.var
    #print("Initial Solution:", bestSol.of)


    weights = dict([(i / 10, 0) for i in range(5, 11)])
    for _ in range(10):
        for i in weights.items():
            newSol, cl = heur.constructBRSol_capacity_given_weight(i[0])  # biased-randomized version of the heuristic
            weights[i[0]] += newSol.of

    weights_sort = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
    a = list(weights_sort.items())


    elapsed = 0.0
    start = time.process_time()
    #iter = 0
    while elapsed < t.Maxtime:
        rand = random.random()
        if rand < 0.7:
            if a[0][0] != 1:
                weight = random.uniform(a[0][0] - 0.05, a[0][0] + 0.05)
            else:
                weight = random.uniform(a[0][0] - 0.05, a[0][0])
        elif rand < 0.9:
            if a[1][0] != 1:
                weight = random.uniform(a[1][0] - 0.05, a[1][0] + 0.05)
            else:
                weight = random.uniform(a[1][0] - 0.05, a[1][0])
        else:
            if a[2][0] != 1:
                weight = random.uniform(a[2][0] - 0.05, a[2][0] + 0.05)
            else:
                weight = random.uniform(a[2][0] - 0.05, a[2][0])

        #iter += 1
        newSol, cl = heur.constructBRSol_capacity_given_weight(weight)  # biased-randomized version of the heuristic
        newSol = tabuSearch_capacity(newSol, cl, t.maxIter, heur)  # Local Search (Tabu Search)
        if newSol.of > bestSol.of:  # Check if the new solution improves the BestSol
            bestSol = newSol.copySol()  # Update Solution
            bestSol.time = elapsed
            #print("New Best Solution:", bestSol.of)

        elapsed = time.process_time() - start
    #print(iter)
    large_simulation = simheuristic(t.long_simulation, var)
    large_simulation.simulation_1(bestSol)
    large_simulation.simulation_2(bestSol, cl)
    return bestSol

'''
Function Main
'''
if __name__ == "__main__":
    tests = readTest("run") # Read the file with the instances to execute

    for t in tests: # Iterate the list with the instances to execute
        random.seed(t.seed)# Set up the seed to used in the execution
        np.random.seed(t.seed)

        path = "CDP/"+t.instName
        inst = Instance(path) #read instance
        if t.inversa != 0:
            inst.b = sum(inst.capacity) * t.inversa
        alpha = 0

        heur = ConstructiveHeuristic(alpha, t.betaBR, t.betaLS, inst, t.weight)
        bestSol, cl = heur.constructBRSol_capacity()  # Greedy Heurístic

        if t.deterministic:
            bestSol = deterministic_multistart(bestSol, t)
        else:
            if t.not_penalization_cost:
                bestSol = stochastic_multistart_simulation(bestSol, t)
            else:
                bestSol = stochastic_multistart(bestSol, t, cl)
        print(t.instName)
        print("of: "+str(bestSol.of))
        #print(len(bestSol.selected))
        '''
        var = 0.1
        small_simulation = simheuristic(100, var)
        small_simulation.simulation_1(bestSol)

        print("Initial Solution:", bestSol.of)

        start = time.process_time()
        elapsed = 0.0
        iter = 0
        elite_simulations = []

        while elapsed < t.Maxtime:
            iter += 1
            newSol, cl = heur.constructBRSol()  # biased-randomized version of the heuristic
            newSol = tabuSearch(newSol, cl, t.maxIter, heur) #Local Search (Tabu Search)
            if newSol.of > bestSol.of: #Check if the new solution improves the BestSol
                small_simulation.simulation_1(newSol)

                if True: #t.penalization_cost
                    if newSol.reliability >= 0.9:
                        elite_simulations.append(newSol)
                        bestSol = newSol.copySol() #Update Solution
                        bestSol.time = elapsed
                        print("New Best Solution:", bestSol.of)

                else:
                    if newSol.stochastic_of >= bestSol.stochastic_of:
                        elite_simulations.append(newSol)
                        bestSol = newSol.copySol() #Update Solution
                        bestSol.time = elapsed
                        print("New Best Solution:", bestSol.of)



            elapsed = time.process_time() - start
        
        
        large_simulation = simheuristic(1000, var)
        elite_simulations.sort(key=lambda x: x.reliability, reverse=True)

        for i in elite_simulations[0:5]:
            large_simulation.simulation_1(i)
        elite_simulations.sort(key=lambda x: x.reliability, reverse=True)
        bestSol = elite_simulations[0]
        '''
        writeData(bestSol, t) #Write solution in the summary Solution File
    sys.exit()






