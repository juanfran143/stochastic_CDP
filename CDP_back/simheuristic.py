import copy

import numpy as np
import random
class simheuristic:

    def __init__(self, simulations, variance):
        self.simulations = simulations
        self.var = variance

    def random_value(self, mean):
        return np.random.lognormal(np.log(mean), self.var)

    def simulation_1(self, solution):
        fail = 0
        capacity = []
        solution.stochastic_of["1"] = solution.of # Optimizar
        for _ in range(self.simulations):
            stochastic_capacity = 0
            for node in solution.selected:
                capacity_node = self.random_value(solution.instance.nodes[node].capacity)
                stochastic_capacity += solution.instance.nodes[node].capacity - (capacity_node - solution.instance.nodes[node].capacity)

            if stochastic_capacity < solution.instance.minCapacity:
                fail += 1

            capacity.append(stochastic_capacity)

        solution.reliability["1"] = (self.simulations-fail)/self.simulations
        solution.total_stochastic_capacity["1"] = np.mean(stochastic_capacity)

    def simulation_2(self, solution):
        fail = 0
        capacity = []
        solution.stochastic_of["2"] = []
        aux_sorted_edges = self.copySortedEdgeList(solution.instance.sortedDistances, solution)
        sortedDistances = solution.instance.sortedDistances
        for _ in range(self.simulations):

            of = solution.of
            stochastic_capacity = 0
            selected_nodes = []
            for i in solution.selected:
                selected_nodes.append(i)

            for node in solution.selected:
                capacity_node = self.random_value(solution.instance.nodes[node].capacity)
                stochastic_capacity += solution.instance.nodes[node].capacity - (
                            capacity_node - solution.instance.nodes[node].capacity)

            if stochastic_capacity < solution.instance.minCapacity:
                fail += 1
                #print(fail)
                sortedDistList = []
                for i in aux_sorted_edges:
                    sortedDistList.append(i)

                while stochastic_capacity < solution.instance.minCapacity*1.25:

                    can = False
                    while not can:  # Induce a BR-behaviour
                        edge = sortedDistList[0]  # select and delect the edge of the position "pos" of the list
                        can = True if (edge.n1.id not in solution.selected or edge.n2.id not in solution.selected) else False
                        if not can:
                            del sortedDistList[sortedDistList.index(edge)]

                    #rand = random.random()
                    # nodeMinCap = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2 # select the node with minimum capacity
                    nodeMinCap = edge.n1 if edge.n1.id not in solution.selected else edge.n2
                    stochastic_capacity += nodeMinCap.capacity
                    nodeMinCap.used = True
                    selected_nodes.append(nodeMinCap.id)

                can = False
                k = 0
                while not can:
                    minEdge = sortedDistances[k]
                    if minEdge.n1.used and minEdge.n2.used:
                        can = True
                    k += 1


                of = minEdge.distance  # Penalitation cost
                #stochastic_capacity = stochastic_capacity#sum(solution.instance.capacity)

            capacity.append(stochastic_capacity)
            solution.stochastic_of["2"].append(of)

        solution.reliability["2"] = (self.simulations - fail) / self.simulations
        solution.total_stochastic_capacity["2"] = np.mean(capacity)
        solution.mean_stochastic_of["2"] = np.mean(solution.stochastic_of["2"])

    def fast_simulation(self, solution, add_node):
        fail = 0

        for _ in range(self.simulations):
            stochastic_capacity = 0
            for node in solution.selected:
                capacity_node = self.random_value(solution.instance.nodes[node].capacity)
                stochastic_capacity += solution.instance.nodes[node].capacity - (
                            capacity_node - solution.instance.nodes[node].capacity)

            capacity_node = self.random_value(solution.instance.nodes[add_node.id].capacity)
            stochastic_capacity += solution.instance.nodes[add_node.id].capacity - (
                    capacity_node - solution.instance.nodes[add_node.id].capacity)

            if stochastic_capacity < solution.instance.minCapacity:
                fail += 1

        p = (self.simulations - fail) / self.simulations
        variance = ((p*(1-p))/self.simulations)**(1/2)
        inf = p-1.96*variance
        sup = p+1.96*self.var
        return (inf, sup)

    def fast_simulation_localsearch(self, solution, add_node, remove_node):
        fail = 0

        for _ in range(self.simulations):
            stochastic_capacity = 0
            for node in solution.selected:
                if node == remove_node:
                    continue
                capacity_node = self.random_value(solution.instance.nodes[node].capacity)
                stochastic_capacity += solution.instance.nodes[node].capacity - (
                            capacity_node - solution.instance.nodes[node].capacity)

            capacity_node = self.random_value(solution.instance.nodes[add_node].capacity)
            stochastic_capacity += solution.instance.nodes[add_node].capacity - (
                    capacity_node - solution.instance.nodes[add_node].capacity)

            if stochastic_capacity < solution.instance.minCapacity:
                fail += 1

        p = (self.simulations - fail) / self.simulations
        return p


    def copySortedEdgeList(self, edgeList,sol):
        newList = []
        for e in edgeList:
            if e.n2.id in sol.nonUsedNodes or e.n1.id in sol.nonUsedNodes:
                newList.append(e)
        return newList