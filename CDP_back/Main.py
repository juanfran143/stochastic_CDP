import sys
from objects import Test, Instance
import random
import os
from Algorithms import Algorithms
import time
import math
from Solution import Solution
from simheuristic import *


def writeData(sol, test):
    fileName = 'output' + os.sep + "ResumeOutputs_DEF_6.txt"

    if not os.path.exists(fileName):
        with open(fileName, "w") as out:
            out.write(
                "Instance\t" + "betaLS\t" + "CostSol\t" + "time\t" + "Capacity\t" + "reliability\t" + "variance\t" + "stochastic_of\t" + "stochastic_capacity\t" + "deterministic\t" + "type_simulation\t" + "inversa\t" + "weight\t" + "seed\n")
    with open(fileName, "a") as out:
        if t.deterministic:
            out.write(test.instName + "\t" + str(test.betaBR) + "\t" + str(sol.of) + "\t" + str(sol.time) + "\t" + str(
                sol.capacity) + "\t" + str(sol.reliability["1"]) + "\t" + str(test.var) + "\t" + str(
                sol.of) + "\t" + str(sol.total_stochastic_capacity["1"]) + "\t" + str(
                t.deterministic) + "\t" + "True" + "\t" + str(t.inversa) + "\t" + str(test.weight) + "\t" + str(
                test.seed) + "\n")
            out.write(
                test.instName + "\t" + str(test.betaBR) + "\t" + str(sol.of) + "\t" + str(sol.time) + "\t" + str(
                    sol.capacity) + "\t" + str(sol.reliability["2"]) + "\t" + str(test.var) + "\t" + str(
                    np.mean(sol.stochastic_of["2"])) + "\t" + str(sol.total_stochastic_capacity["2"]) + "\t" + str(
                    test.deterministic) + "\t" + "False" + "\t" + str(t.inversa) + "\t" + str(test.weight) + "\t" + str(
                    test.seed) + "\n")
        else:
            if t.not_penalization_cost:
                out.write(test.instName + "\t" + str(test.betaBR) + "\t" + str(sol.of) + "\t" + str(
                    sol.time) + "\t" + str(sol.capacity) + "\t" + str(sol.reliability["1"]) + "\t" + str(
                    test.var) + "\t" + str(sol.of) + "\t" + str(sol.total_stochastic_capacity["1"]) + "\t" + str(
                    t.deterministic) + "\t" + str(test.not_penalization_cost) + "\t" + str(test.inversa) + "\t" + str(
                    test.weight) + "\t" + str(test.seed) + "\n")
            else:
                out.write(
                    test.instName + "\t" + str(test.betaBR) + "\t" + str(sol.of) + "\t" + str(
                        sol.time) + "\t" + str(
                        sol.capacity) + "\t" + str(sol.reliability["2"]) + "\t" + str(test.var) + "\t" + str(
                        np.mean(sol.stochastic_of["2"])) + "\t" + str(
                        sol.total_stochastic_capacity["2"]) + "\t" + str(test.deterministic) + "\t" + str(
                        test.not_penalization_cost) + "\t" + str(test.inversa) + "\t" + str(test.weight) + "\t" + str(
                        test.seed) + "\n")


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

def writeData(sol, test):
        fileName = 'output' + os.sep + "ResumeOutputs.txt"

        if not os.path.exists(fileName):
            with open(fileName, "w") as out:
                out.write(
                    "Instance\t" + "betaLS\t" + "CostSol\n")
        with open(fileName, "a") as out:
            out.write(test.instName + "\t" + "\t" + str(sol.of) + "\t" + str(sol.time) + "\n")
'''


def stochastic_multistart(bestSol, alg, t: Instance):
    var = t.var
    small_simulation = simheuristic(t.short_simulation, var)

    if t.not_penalization_cost:
        small_simulation.simulation_1(bestSol)
    else:
        small_simulation.simulation_2(bestSol)

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
        for i in inst.nodes:
            i.used = False
        newSol = Solution(inst)
        newSol = alg.ConstructiveHeuristic_without_last_move(newSol, True)  # BRConstructive Heurístic
        newSol = alg.localSearch(newSol)
        newSol.time = time.process_time() - start
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
                small_simulation.simulation_2(newSol)
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
    if not enter and t.not_penalization_cost and len(elite_enter_simulations) > 0:
        for i in elite_enter_simulations:
            large_simulation.simulation_1(i)
        elite_enter_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
        return elite_enter_simulations[0]

    for i in elite_simulations:
        if t.not_penalization_cost:
            large_simulation.simulation_1(i)
        else:
            large_simulation.simulation_2(i)

    if t.not_penalization_cost:
        elite_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
    else:
        elite_simulations.sort(key=lambda x: x.stochastic_of["2"], reverse=True)

    return elite_simulations[0]

def stochastic_multistart_safety(bestSol, alg, t: Instance):
    var = t.var
    small_simulation = simheuristic(t.short_simulation, var)

    if t.not_penalization_cost:
        small_simulation.simulation_1(bestSol)
    else:
        small_simulation.simulation_2(bestSol)

    # print("Initial Solution:", bestSol.of)

    elapsed = 0.0
    iter = 0
    elite_simulations = []
    elite_enter_simulations = []
    elite_simulations.append(bestSol)
    start = time.process_time()
    bestSol_axu = bestSol.copySol()

    enter = False
    safety = 0
    while elapsed < t.Maxtime:
        iter += 1
        for i in inst.nodes:
            i.used = False

        if t.not_penalization_cost:

            newSol = Solution(inst)
            newSol = alg.ConstructiveHeuristic_without_last_move_safety(newSol, safety)  # BRConstructive Heurístic
            newSol = alg.localSearch_safety(newSol, safety)

        else:
            newSol = Solution(inst)
            newSol = alg.ConstructiveHeuristic_without_last_move(newSol, True)  # BRConstructive Heurístic
            newSol = alg.localSearch(newSol)

        newSol.time = time.process_time() - start

        if newSol.of > bestSol.of:  # Check if the new solution improves the BestSol
            if t.not_penalization_cost:
                iter += 1
                if iter >= 20:
                    iter = 0
                    safety += 0.01
                small_simulation.simulation_1(newSol)
                print(newSol.reliability["1"])
                if newSol.reliability["1"] >= t.delta:
                    iter = 0
                    enter = True
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)

                elif not enter and newSol.reliability["1"] >= bestSol_axu.reliability["1"]:
                    bestSol_axu = newSol.copySol()  # Update Solution
                    bestSol_axu.time = elapsed
                    elite_enter_simulations.append(bestSol_axu)
            else:
                small_simulation.simulation_2(newSol)
                # print("best: "+ str(bestSol.mean_stochastic_of["2"]) + " New_sol: " + str(newSol.mean_stochastic_of["2"])+ " tiempo:"+str(elapsed))
                if newSol.mean_stochastic_of["2"] >= bestSol.mean_stochastic_of["2"]:
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)
                    # print("New Best Time:", elapsed)
                    # print("New Best Solution:", bestSol.of)

        elapsed = time.process_time() - start
    # print(iter)
    large_simulation = simheuristic(t.long_simulation, var)
    if not enter and t.not_penalization_cost and len(elite_enter_simulations) > 0:
        for i in elite_enter_simulations:
            large_simulation.simulation_1(i)
        elite_enter_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
        return elite_enter_simulations[0]

    for i in elite_simulations:
        if t.not_penalization_cost:
            large_simulation.simulation_1(i)
        else:
            large_simulation.simulation_2(i)

    if t.not_penalization_cost:
        elite_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
    else:
        elite_simulations.sort(key=lambda x: x.stochastic_of["2"], reverse=True)

    return elite_simulations[0]

def stochastic_multistart_safety(bestSol, alg, t: Instance):
    var = t.var
    small_simulation = simheuristic(t.short_simulation, var)

    if t.not_penalization_cost:
        small_simulation.simulation_1(bestSol)
    else:
        small_simulation.simulation_2(bestSol)

    # print("Initial Solution:", bestSol.of)

    elapsed = 0.0
    iter = 0
    elite_simulations = []
    elite_enter_simulations = []
    elite_simulations.append(bestSol)
    start = time.process_time()
    bestSol_axu = bestSol.copySol()

    enter = False
    safety = 0
    while elapsed < t.Maxtime:
        iter += 1
        for i in inst.nodes:
            i.used = False

        if t.not_penalization_cost:

            newSol = Solution(inst)
            newSol = alg.ConstructiveHeuristic_without_last_move_safety(newSol, safety)  # BRConstructive Heurístic
            newSol = alg.localSearch_safety(newSol, safety)

        else:
            newSol = Solution(inst)
            newSol = alg.ConstructiveHeuristic_without_last_move(newSol, True)  # BRConstructive Heurístic
            newSol = alg.localSearch(newSol)

        newSol.time = time.process_time() - start

        if newSol.of > bestSol.of:  # Check if the new solution improves the BestSol
            if t.not_penalization_cost:
                iter += 1
                if iter >= 20:
                    iter = 0
                    safety += 0.01
                small_simulation.simulation_1(newSol)
                print(newSol.reliability["1"])
                if newSol.reliability["1"] >= t.delta:
                    iter = 0
                    enter = True
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)

                elif not enter and newSol.reliability["1"] >= bestSol_axu.reliability["1"]:
                    bestSol_axu = newSol.copySol()  # Update Solution
                    bestSol_axu.time = elapsed
                    elite_enter_simulations.append(bestSol_axu)
            else:
                small_simulation.simulation_2(newSol)
                # print("best: "+ str(bestSol.mean_stochastic_of["2"]) + " New_sol: " + str(newSol.mean_stochastic_of["2"])+ " tiempo:"+str(elapsed))
                if newSol.mean_stochastic_of["2"] >= bestSol.mean_stochastic_of["2"]:
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)
                    # print("New Best Time:", elapsed)
                    # print("New Best Solution:", bestSol.of)

        elapsed = time.process_time() - start
    # print(iter)
    large_simulation = simheuristic(t.long_simulation, var)
    if not enter and t.not_penalization_cost and len(elite_enter_simulations) > 0:
        for i in elite_enter_simulations:
            large_simulation.simulation_1(i)
        elite_enter_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
        return elite_enter_simulations[0]

    for i in elite_simulations:
        if t.not_penalization_cost:
            large_simulation.simulation_1(i)
        else:
            large_simulation.simulation_2(i)

    if t.not_penalization_cost:
        elite_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
    else:
        elite_simulations.sort(key=lambda x: x.stochastic_of["2"], reverse=True)

    return elite_simulations[0]

def stochastic_multistart_simulation(bestSol, alg, t: Instance):
    var = t.var
    small_simulation = simheuristic(t.short_simulation, var)

    if t.not_penalization_cost:
        for i in inst.nodes:
            i.used = False
        bestSol = Solution(inst)
        bestSol = alg.ConstructiveHeuristic_without_last_move_simulation(bestSol, simulation=simheuristic(20, var),alpha=t.delta)  # BRConstructive Heurístic
        #bestSol = alg.localSearch_simulation(bestSol, simulation=simheuristic(20, var), alpha=t.delta)

    if t.not_penalization_cost:
        small_simulation.simulation_1(bestSol)
    else:
        small_simulation.simulation_2(bestSol)

    # print("Initial Solution:", bestSol.of)

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
        #print(iter)
        for i in inst.nodes:
            i.used = False

        if t.not_penalization_cost:
            #mini_simulation = simheuristic(20, var)
            newSol = Solution(inst)
            newSol = alg.ConstructiveHeuristic_without_last_move_simulation(newSol, simulation=simheuristic(20, var), alpha=t.delta)  # BRConstructive Heurístic
            #newSol = alg.localSearch_simulation(newSol, simulation=simheuristic(20, var), alpha=t.delta)
            #small_simulation.simulation_1(newSol)
            #print(newSol.reliability["1"])
        else:
            newSol = Solution(inst)
            newSol = alg.ConstructiveHeuristic_without_last_move(newSol, True)  # BRConstructive Heurístic
            newSol = alg.localSearch(newSol)

        newSol.time = time.process_time() - start

        if newSol.of > bestSol.of:  # Check if the new solution improves the BestSol
            if t.not_penalization_cost:
                small_simulation.simulation_1(newSol)

                if newSol.reliability["1"] >= t.delta:
                    enter = True
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)

                elif not enter and newSol.reliability["1"] >= bestSol_axu.reliability["1"]:
                    bestSol_axu = newSol.copySol()  # Update Solution
                    bestSol_axu.time = elapsed
                    elite_enter_simulations.append(bestSol_axu)
            else:
                small_simulation.simulation_2(newSol)
                # print("best: "+ str(bestSol.mean_stochastic_of["2"]) + " New_sol: " + str(newSol.mean_stochastic_of["2"])+ " tiempo:"+str(elapsed))
                if newSol.mean_stochastic_of["2"] >= bestSol.mean_stochastic_of["2"]:
                    bestSol = newSol.copySol()  # Update Solution
                    bestSol.mean_stochastic_of["2"] = newSol.mean_stochastic_of["2"]
                    bestSol.reliability = newSol.reliability
                    bestSol.time = elapsed
                    elite_simulations.append(bestSol)
                    # print("New Best Time:", elapsed)
                    # print("New Best Solution:", bestSol.of)

        elapsed = time.process_time() - start
    # print(iter)
    large_simulation = simheuristic(t.long_simulation, var)
    if not enter and t.not_penalization_cost and len(elite_enter_simulations) > 0:
        for i in elite_enter_simulations:
            large_simulation.simulation_1(i)
        elite_enter_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
        return elite_enter_simulations[0]

    for i in elite_simulations:
        if t.not_penalization_cost:
            large_simulation.simulation_1(i)
        else:
            large_simulation.simulation_2(i)

    if t.not_penalization_cost:
        elite_simulations.sort(key=lambda x: x.reliability["1"], reverse=True)
    else:
        elite_simulations.sort(key=lambda x: x.stochastic_of["2"], reverse=True)

    return elite_simulations[0]

def deterministic_multistart(bestSol: Solution, alg: Algorithms, t: Instance)->Solution:
    var = t.var
    #print("Initial Solution:", bestSol.of)

    bestSol = alg.ConstructiveHeuristic(bestSol, False)  # Constructive Heurístic
    bestSol = alg.localSearch(bestSol)
    # print(bestSol.of)
    start = time.process_time()
    while time.process_time() - start < t.Maxtime:
        for i in inst.nodes:
            i.used = False
        newSol = Solution(inst)
        newSol = alg.ConstructiveHeuristic_without_last_move(newSol, True)  # BRConstructive Heurístic
        newSol = alg.localSearch(newSol)
        newSol.time = time.process_time() - start
        if newSol.of > bestSol.of:
            bestSol = newSol.copySol()
            bestSol.time = newSol.time

    #print(iter)
    large_simulation = simheuristic(t.long_simulation, var)
    large_simulation.simulation_1(bestSol)
    large_simulation.simulation_2(bestSol)
    return bestSol


'''
Function Main
'''
if __name__ == "__main__":
    tests = readTest("test2run") # Read the file with the instances to execute
    for t in tests: # Iterate the list with the instances to execute
        path = "CDP/"+t.instName
        print(path)
        inst = Instance()
        inst.readInstance(path) #read instance
        if t.inversa != 0:
            inst.minCapacity = sum([i.capacity for i in inst.nodes]) * t.inversa

        elapsed = 0.0
        iter = 0  # variable to count the total number of iterations
        p = 0  # percentage of nodes to destroy of the solution
        credit = 0 #Used to accept solutions that does not improved the bestSol (e.g Simulated aneeling mechanish)
        random.seed(t.seed)  # Set up the seed to used in the execution

        #Phase 1: Create a Initial Solution
        alg = Algorithms(0.2, inst) #create the object with the heuristic functions
        bestSol = Solution(inst) #Create a Solution with all the nodes open
        bestSol = alg.ConstructiveHeuristic(bestSol, False)

        if t.deterministic:
            bestSol = deterministic_multistart(bestSol, alg, t)
        else:
            bestSol = stochastic_multistart_simulation(Solution(inst), alg, t)
        #--------------->Multistart to test the BRHeuristic<----------------

        print("of: "+str(bestSol.of))
        print("of stochastic: " + str(np.mean(bestSol.stochastic_of["2"])))
        writeData(bestSol, t)

        '''
        exit(-1)
        #fin testtttt
        exit(-1) #Salgo para no entrar en el ILS por ahora



        baseSol = bestSol.copySol() #Assign this solution as baseSol
        print("Initial Solution:", bestSol.of)


        #Phase2: ILS PROCEDURE (The objective is to improve the initial Solution)
        start = time.process_time()
        while elapsed < t.Maxtime: #Iterate until the computational time reach the maxTime
            if p < 100:
                p += 1 #increse the percentage to destroy the solution
            newSol = alg.pertubation(baseSol,p) #Pertubation Method (Takes baseSol and adds p% of non-in-Sol nodes to BaseSol)
            newSol =  alg.localSearch(newSol) # <- OJOOOOOO!!! : Puedo tener 2 aristas de misma distancia!!!!!! <----- Desempatar por la min.  cap. de nodos
            delta = newSol.of - baseSol.of
            if delta > 0: #Check if the new solution improves the baseSol
                credit = delta #set up the credit.
                baseSol = newSol.copySol()  # Update baseSol
                if newSol.of > bestSol.of:  #Check if the new solution improves the BestSol
                    bestSol = newSol.copySol() #Update bestSol
                    bestSol.time = elapsed
                    print("New Best Solution:", bestSol.of)
                    p = 0 #reset to the percetange o nodes to be deleted
            else: #If the baseSol is not improved the credit is checked
                if math.fabs(delta) < credit: #accept a worse solution as baseSol
                    baseSol = newSol.copySol()
                    p = 0  # reset to the percetange o nodes to be deleted
            elapsed = time.process_time() - start
            iter += 1
        writeData(bestSol, t)  # Write solution in the summary Solution File
        print(iter)
    sys.exit()
'''


