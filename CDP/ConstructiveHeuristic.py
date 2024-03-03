import math
from Solution import Solution
from objects import Candidate, Candidate_capacity
import random


class ConstructiveHeuristic:
    def __init__(self,alpha,beta,betaLS,inst, weight):
        self.alpha = alpha
        self.firstEdge = 0
        self.beta = beta
        self.betaLS = betaLS
        self.instance = inst

        self.weight = weight


    #original Grasp Heuristic
    #Constructive heuristic (Deterministic Version)
    def constructSolution(self):
        sol = Solution(self.instance)
        edge = self.instance.sortedDistances[self.firstEdge]
        sol.add(edge.v1)
        sol.add(edge.v2)
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        cl = self.createCL(sol)
        realAlpha = self.alpha if self.alpha >= 0 else random.random()
        while(not sol.isFeasible()):
            distanceLimit = cl[0].cost - (realAlpha * cl[len(cl)-1].cost)
            i = 0
            maxCap = 0
            vWithMaxCap = -1
            while i < len(cl) and (cl[i].cost >= distanceLimit):
                v = cl[i].v
                vCap = self.instance.capacity[v]
                if vCap > maxCap:
                    maxCap = vCap
                    vWithMaxCap = i
                i+=1
            c = cl.pop(vWithMaxCap)
            sol.add(c.v)

            if c.cost < sol.of:
                sol.updateOF(c.v, c.closestV, c.cost)

            # Debug
            #if sol.getEvalComplete() != sol.of:
            #    print("MAL: " + str(sol.getEvalComplete()) + " vs " + str(sol.of))

            self.updateCL(sol, cl, c.v)
        return sol



    def constructSolution_capacity(self, weight):
        sol = Solution(self.instance)
        edge = self.instance.sortedDistances[self.firstEdge]
        sol.add(edge.v1)
        sol.add(edge.v2)
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        self.weight = weight
        cl = self.createCL_capacity(sol)
        realAlpha = self.alpha if self.alpha >= 0 else random.random()
        while(not sol.isFeasible()):
            distanceLimit = cl[0].cost - (realAlpha * cl[len(cl)-1].cost)
            i = 0
            maxCap = 0
            vWithMaxCap = -1
            while i < len(cl) and (cl[i].cost >= distanceLimit):
                v = cl[i].v
                vCap = self.instance.capacity[v]
                if vCap > maxCap:
                    maxCap = vCap
                    vWithMaxCap = i
                i+=1
            c = cl.pop(vWithMaxCap)
            sol.add(c.v)

            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)

            self.updateCL_capacity(sol, cl, c.v)
        return sol


    #BR-Heuristic
    def constructBRSol(self):
        sol = Solution(self.instance)
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        edge = self.instance.sortedDistances[pos]
        sol.add(edge.v1)
        sol.add(edge.v2)
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        cl = self.createCL(sol)
        while(not sol.isFeasible()):
            pos = self.getRandomPosition(len(cl), random, self.beta)
            v = cl[pos].v
            c = cl.pop(pos)
            sol.add(c.v)

            if c.cost < sol.of:
                sol.updateOF(c.v, c.closestV, c.cost)

            self.updateCL(sol, cl, c.v)
        return sol, cl

    def constructBRSol_capacity(self):
        sol = Solution(self.instance)
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        edge = self.instance.sortedDistances[pos]
        sol.add(edge.v1)
        sol.add(edge.v2)
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        self.weight = random.uniform(0.6, 0.9)  # Random entre 0.8 y 0.9
        cl = self.createCL_capacity(sol)
        while (not sol.isFeasible()):
            pos = self.getRandomPosition(len(cl), random, self.beta)
            v = cl[pos].v
            c = cl.pop(pos)
            sol.add(c.v)
            if self.max_capacity < self.instance.capacity[c.v]:
                self.max_capacity = self.instance.capacity[c.v]
            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)

            self.updateCL_capacity(sol, cl, c.v)
        return sol, cl

    def constructBRSol_capacity_simulation(self, simulation, delta):
        sol = Solution(self.instance)
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        edge = self.instance.sortedDistances[pos]
        sol.add(edge.v1)
        sol.add(edge.v2)
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        self.weight = random.uniform(0.6, 0.9)  # Random entre 0.6 y 0.9
        cl = self.createCL_capacity(sol)

        lower, upper = simulation.fast_simulation(sol)

        while(lower < delta):
            pos = self.getRandomPosition(len(cl), random, self.beta)
            v = cl[pos].v
            c = cl.pop(pos)
            sol.add(c.v)
            if self.max_capacity < self.instance.capacity[c.v]:
                self.max_capacity = self.instance.capacity[c.v]
            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)

            self.updateCL_capacity(sol, cl, c.v)
            lower, upper = simulation.fast_simulation(sol)
        return sol, cl

    def constructBRSol_capacity_given_weight(self, weight):
        sol = Solution(self.instance)
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        edge = self.instance.sortedDistances[pos]
        sol.add(edge.v1)
        sol.add(edge.v2)
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        self.weight = weight
        cl = self.createCL_capacity(sol)
        while(not sol.isFeasible()):
            pos = self.getRandomPosition(len(cl), random, self.beta)
            v = cl[pos].v
            c = cl.pop(pos)
            sol.add(c.v)
            if self.max_capacity < self.instance.capacity[c.v]:
                self.max_capacity = self.instance.capacity[c.v]
            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)

            self.updateCL_capacity(sol, cl, c.v)
        return sol, cl

    def constructBRSol_capacity_ajusted(self, ajusted):
        sol = Solution(self.instance)
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        edge = self.instance.sortedDistances[pos]
        sol.add(edge.v1)
        sol.add(edge.v2)
        sol.updateOF(edge.v1, edge.v2, edge.distance)

        random_number = random.random()
        before = 0
        selected = list(ajusted.keys())[-1]
        for i in ajusted.items():
            if random_number <= i[1] + before:
                selected = i[0]
                break
            else:
                before += i[1]


        self.weight = random.uniform(selected[0], selected[1])  # Random entre 0.8 y 0.9
        cl = self.createCL_capacity(sol)
        while(not sol.isFeasible()):
            pos = self.getRandomPosition(len(cl), random, self.beta)
            v = cl[pos].v
            c = cl.pop(pos)
            sol.add(c.v)
            if self.max_capacity < self.instance.capacity[c.v]:
                self.max_capacity = self.instance.capacity[c.v]
            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)

            self.updateCL_capacity(sol, cl, c.v)
        return sol, cl, selected



    def createCL(self, sol):
        instance = sol.instance
        n = instance.n
        cl = [] #Candidate List of nodes
        for v in range(0,n):
            if v in sol.selected:
                continue
            vMin, minDist = sol.distanceTo(v)
            c = Candidate(v, vMin, minDist)
            cl.append(c)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos
        return cl

    def createCL_capacity(self, sol):
        instance = sol.instance
        n = instance.n
        cl = [] #Candidate List of nodes
        nodes = []
        capacity = []
        for v in range(0,n):
            if v in sol.selected:
                capacity.append(instance.capacity[v])
                continue
            vMin, minDist = sol.distanceTo(v)
            #cost = minDist / self.max_min_dist * self.weight + instance.capacity[v] / self.max_capacity * (
            #            1 - self.weight)
            c = Candidate(v, vMin, minDist)
            nodes.append(c)

        self.max_min_dist = max([x.cost for x in nodes])
        self.max_capacity = max(capacity)

        for i in nodes:
            cost = i.cost / self.max_min_dist * self.weight + instance.capacity[i.v] / self.max_capacity * (1 - self.weight)
            c = Candidate_capacity(i.v, i.closestV, i.cost, cost)
            cl.append(c)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos

        return cl


    def insertNodeToCL(self,cl,sol,v):
        vMin, minDist = sol.distanceTo(v)
        c = Candidate(v, vMin, minDist)
        cl.append(c)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos
        return cl

    def insertNodeToCL_capacity(self,cl,sol,v):
        capacity_v = self.instance.capacity[v]
        if self.max_capacity < capacity_v:
            self.max_capacity = capacity_v

        vMin, minDist = sol.distanceTo(v)

        if self.max_min_dist < minDist:
            self.max_min_dist = minDist

        if self.max_min_dist!= 0:
            cost = minDist / self.max_min_dist * self.weight + capacity_v / self.max_capacity * (1 - self.weight)
        else:
            cost = capacity_v / self.max_capacity * (1 - self.weight)

        c = Candidate_capacity(v, vMin, minDist, cost)
        cl.append(c)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos
        return cl

    #Used in the LS to include new nodes in the solution
    def partialReconstruction(self,sol,cl):
        while (not sol.isFeasible()):
            pos = self.getRandomPosition(len(cl), random,self.betaLS)
            c = cl.pop(pos)
            sol.add(c.v)
            self.updateCL(sol, cl, c.v)
        return sol


    def partialReconstruction_capacity(self,sol,cl):

        while (not sol.isFeasible()):
            pos = self.getRandomPosition(len(cl), random, self.betaLS)
            c = cl.pop(pos)
            sol.add(c.v)
            self.updateCL_capacity(sol, cl, c.v)
        return sol


    def partialReconstruction_capacity_simulation(self,sol,cl, simulation, delta):
        lower, upper = simulation.fast_simulation(sol)
        while lower < delta:
            pos = self.getRandomPosition(len(cl), random, self.betaLS)
            c = cl.pop(pos)
            sol.add(c.v)
            self.updateCL_capacity(sol, cl, c.v)
            lower, upper = simulation.fast_simulation(sol)
        return sol


    def updateCL(self,sol,cl,lastAdded):
        instance = sol.instance
        for c in cl:
            dToLast = instance.distance[lastAdded][c.v]
            if dToLast < c.cost:
                c.cost = dToLast
                c.closestV = lastAdded
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos

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
                c.cost = c.dist_min / self.max_min_dist * self.weight + instance.capacity[c.v] / self.max_capacity * (1 - self.weight)
            else:
                c.cost = instance.capacity[c.v] / self.max_capacity * (1 - self.weight)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos




    def recalculateCL(self,sol,cl,lastDrop):
        instance = sol.instance
        for c in cl:
            if lastDrop == c.closestV:
                vMin, minDist = sol.distanceTo(c.v)
                c.cost = minDist
                c.closestV = vMin
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos


    def recalculateCL_capacity(self,sol,cl,lastDrop):
        instance = sol.instance
        for c in cl:
            if lastDrop == c.closestV:
                vMin, minDist = sol.distanceTo(c.v)
                c.dist_min = minDist
                c.closestV = vMin

        self.max_min_dist = max([x.dist_min for x in cl])
        if self.max_capacity == instance.capacity[lastDrop]:
            self.max_capacity = max([instance.capacity[i] for i in sol.selected])

        for c in cl:
            if self.max_min_dist != 0:
                c.cost = c.dist_min / self.max_min_dist * self.weight + instance.capacity[c.v] / self.max_capacity * (
                    1 - self.weight)
            else:
                c.cost = instance.capacity[c.v] / self.max_capacity * (1 - self.weight)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos




    def getRandomPosition(self,size, random,beta):
        index = int(math.log(random.random()) / math.log(1 - beta))
        index = index % size
        return index
