import copy

import numpy as np
from ConstructiveHeuristic import *

class simheuristic:

    def __init__(self, simulations, variance):
        self.simulations = simulations
        self.var = variance


    def random_value(self, mean):
        return np.random.lognormal(np.log(mean), self.var)

    def simulation_1(self, solution: Solution):
        fail = 0
        capacity = []
        solution.stochastic_of["1"] = solution.of # Optimizar
        for _ in range(self.simulations):
            stochastic_capacity = 0
            for node in solution.selected:
                capacity_node = self.random_value(solution.instance.capacity[node])
                stochastic_capacity += solution.instance.capacity[node] - (capacity_node - solution.instance.capacity[node])

            if stochastic_capacity < solution.instance.b:
                fail += 1

            capacity.append(stochastic_capacity)

        solution.reliability["1"] = (self.simulations-fail)/self.simulations
        solution.total_stochastic_capacity["1"] = np.mean(stochastic_capacity)

    def simulation_2(self, solution: Solution, cl):
        fail = 0

        capacity = []
        solution.stochastic_of["2"] = []
        for _ in range(self.simulations):
            aux_solution = Solution(solution.instance)
            aux_solution.of = solution.of
            aux_solution.of = solution.vMin1
            aux_solution.of = solution.vMin2
            of = solution.of
            stochastic_capacity = 0
            for node in solution.selected:
                capacity_node = self.random_value(solution.instance.capacity[node])
                stochastic_capacity += solution.instance.capacity[node] - (
                            capacity_node - solution.instance.capacity[node])

            if stochastic_capacity < solution.instance.b:
                fail += 1
                #TODO
                #I have changed it
                #of = solution.instance.sortedDistances[-1].distance  # Penalitation cost
                i = 0
                while stochastic_capacity < solution.instance.b:
                    v = cl[i].v
                    stochastic_capacity += solution.instance.capacity[v]
                    if of > cl[i].dist_min:
                        #sol.updateOF(c.v, c.closestV, c.dist_min)
                        of = cl[i].dist_min
                    aux_solution.updateOF(cl[i].v, cl[i].closestV, cl[i].dist_min)
                    self.updateCL_capacity(aux_solution, cl, cl[i].v)
                    i += 1
                    #c = cl.pop(vWithMaxCap)
                #of = (solution.instance.sortedDistances[0].distance-solution.instance.sortedDistances[-1].distance)/4
                stochastic_capacity = sum(solution.instance.capacity)

            #print("OF: "+str(of))
            capacity.append(stochastic_capacity)
            solution.stochastic_of["2"].append(of)

        solution.reliability["2"] = (self.simulations - fail) / self.simulations
        solution.total_stochastic_capacity["2"] = np.mean(capacity)
        solution.mean_stochastic_of["2"] = np.mean(solution.stochastic_of["2"])


    def fast_simulation(self, solution: Solution):
        fail = 0
        for _ in range(self.simulations):
            stochastic_capacity = 0
            for node in solution.selected:
                capacity_node = self.random_value(solution.instance.capacity[node])
                stochastic_capacity += solution.instance.capacity[node] - (
                            capacity_node - solution.instance.capacity[node])

            if stochastic_capacity < solution.instance.b:
                fail += 1

        p = (self.simulations - fail) / self.simulations
        variance = ((p*(1-p))/self.simulations)**(1/2)
        inf = p-1.96*variance
        sup = p+1.96*self.var
        return (inf, sup)


    def updateCL_capacity(self, sol, cl, lastAdded):
        instance = sol.instance
        for c in cl:
            dToLast = instance.distance[lastAdded][c.v]
            #cost = dToLast/self.max_min_dist * self.weight + instance.capacity[c.v]/self.max_capacity * (1-self.weight)
            if dToLast < c.dist_min:
                c.dist_min = dToLast
                c.closestV = lastAdded

        self.max_min_dist = max([x.dist_min for x in cl])
        for c in cl:

            if self.max_min_dist != 0:
                c.cost = c.dist_min / self.max_min_dist * 0.8 + instance.capacity[c.v] / max(instance.capacity) * (1 - 0.8)
            else:
                c.cost = instance.capacity[c.v] / max(instance.capacity) * (1 - 0.8)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos