import math
from Solution import Solution
import random
import sys
import copy
import numpy as np
from simheuristic import simheuristic

class Algorithms:
    def __init__(self,beta,inst):
        self.beta = beta
        #self.safety = 0
        self.instance = inst


    '''
        The heuristic considers that initially the solution contains all the nodes, 
        and iteratively closes nodes while the capacity of the solution is higher to 
        the minimum capacity.
        Parameters:
        Sol: Solution
        isRandom: induce a BR behaviour (False-> Gready Behaviour; True-> BR Behaviour)
    '''
    def ConstructiveHeuristic(self, sol, isRandom):
            sol.of = 0
            sortedDistList = self.copySortedEdgeList(self.instance.sortedDistances, sol) #Copy the edge list
            first_move = True
            while (len(sortedDistList) != 0):
                #if not first_move:
                #    self.beta = 0.99
                #pos = 0 #Iterate the list in a greedy way
                #if isRandom:
                can = False
                while not can:
                    pos = self.getRandomPosition(len(sortedDistList), self.beta) #Induce a BR-behaviour
                    edge = sortedDistList[pos] #select and delect the edge of the position "pos" of the list
                    can = True if not edge.n1.used and not edge.n2.used else False
                    if not can:
                        del sortedDistList[sortedDistList.index(edge)]

                rand = random.random()
                #nodeMinCap = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2 # select the node with minimum capacity
                nodeMinCap = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2

                if (sol.capacity - nodeMinCap.capacity) >= self.instance.minCapacity: #I can delete the node because I have not violate the demand
                    first_move = False
                    nodeMinCap.used = True
                    del sol.selected[sol.selected.index(nodeMinCap.id)] #delete the node with minimum capacity of the solution
                    sol.nonUsedNodes.append(nodeMinCap.id)
                    sol.capacity -= nodeMinCap.capacity #substract the capacity of the node of the solution.
                    #for e in nodeMinCap.associatedEdges: #update the edges involves in the deleted node
                    #    if e in sortedDistList and e!=edge:
                    #        del sortedDistList[sortedDistList.index(e)] #borro los edges asociados
                else: #I can not delete the node due to capacity constraints.

                    can = False
                    k = 0
                    while not can:
                        minEdge = sortedDistList[k]
                        if not minEdge.n1.used and not minEdge.n2.used:
                            can = True
                        k += 1

                    sol.of = minEdge.distance
                    sol.vMin1 = minEdge.n1.id
                    sol.vMin2 = minEdge.n2.id

                    if not first_move:
                        node_before = 0
                        vmin1 = True
                        if discarted_node.id == sol.vMin1:
                            node_before = sol.vMin2
                            data = np.array([(i.distance,i) for i in discarted_node.associatedEdges if (i.n1.id in sol.selected or i.n2.id in sol.selected) and i.n1.id != sol.vMin2 and i.n2.id != sol.vMin2])
                            sortIndices = np.argsort(data[:, 0])
                            min_descarted_edge = data[sortIndices[0],:]
                        elif discarted_node.id == sol.vMin2:
                            vmin1 = False
                            node_before = sol.vMin1
                            data = np.array([(i.distance,i) for i in discarted_node.associatedEdges if (i.n1.id in sol.selected or i.n2.id in sol.selected) and i.n1.id != sol.vMin1 and i.n2.id != sol.vMin1])
                            sortIndices = np.argsort(data[:, 0])
                            min_descarted_edge = data[sortIndices[0],:]
                        else:
                            min_descarted_edge = (minEdge.distance, minEdge)

                        introduce = min_descarted_edge[1].n1.id if sol.vMin1 != min_descarted_edge[1].n1 and sol.vMin2 != min_descarted_edge[1].n1 else min_descarted_edge[1].n2.id

                        if sol.capacity + self.instance.nodes[introduce].capacity - self.instance.nodes[node_before].capacity:
                            of = sol.getEvalComplete()
                            if of > minEdge.distance:
                                sol.of = of

                                if vmin1:
                                    sol.vMin2 = introduce
                                else:
                                    sol.vMin1 = introduce

                                sol.selected.append(introduce)
                                sol.selected.remove(node_before)

                                sol.nonUsedNodes.append(node_before)
                    # min_arista = min(non_discarted.associatedEdges)
                    # min([i.distance for i in non_discarted.associatedEdges if i.id in sol.selected.id and i.id != sol.vMin1])
                    # if minEdge <

                    break
                #copy_edge = edge
                #discarted_node = edge.n2 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n1
                #selected = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2
                discarted_node = edge.n2 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n1
                #selected = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2
                sortedDistList.remove(edge)


            #if sol.of != sol.getEvalComplete(): #just to debugging purposes!!!!
            #    print("Erorrrrrr",sol.of,sol.getEvalComplete())
            return sol

    def ConstructiveHeuristic_without_last_move(self, sol, isRandom):
            sol.of = 0
            sortedDistList = self.copySortedEdgeList(self.instance.sortedDistances, sol) #Copy the edge list
            while (len(sortedDistList) != 0):
                #if not first_move:
                #    self.beta = 0.99
                #pos = 0 #Iterate the list in a greedy way
                #if isRandom:
                can = False
                while not can:
                    pos = self.getRandomPosition(len(sortedDistList), self.beta) #Induce a BR-behaviour
                    edge = sortedDistList[pos] #select and delect the edge of the position "pos" of the list
                    can = True if not edge.n1.used and not edge.n2.used else False
                    if not can:
                        del sortedDistList[sortedDistList.index(edge)]

                rand = random.random()
                #nodeMinCap = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2 # select the node with minimum capacity
                nodeMinCap = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2

                if (sol.capacity - nodeMinCap.capacity) >= self.instance.minCapacity: #I can delete the node because I have not violate the demand
                    nodeMinCap.used = True
                    del sol.selected[sol.selected.index(nodeMinCap.id)] #delete the node with minimum capacity of the solution
                    sol.nonUsedNodes.append(nodeMinCap.id)
                    sol.capacity -= nodeMinCap.capacity #substract the capacity of the node of the solution.
                    #for e in nodeMinCap.associatedEdges: #update the edges involves in the deleted node
                    #    if e in sortedDistList and e!=edge:
                    #        del sortedDistList[sortedDistList.index(e)] #borro los edges asociados
                else: #I can not delete the node due to capacity constraints.

                    can = False
                    k = 0
                    while not can:
                        minEdge = sortedDistList[k]
                        if not minEdge.n1.used and not minEdge.n2.used:
                            can = True
                        k += 1

                    sol.of = minEdge.distance
                    sol.vMin1 = minEdge.n1.id
                    sol.vMin2 = minEdge.n2.id

                    # min_arista = min(non_discarted.associatedEdges)
                    # min([i.distance for i in non_discarted.associatedEdges if i.id in sol.selected.id and i.id != sol.vMin1])
                    # if minEdge <

                    break
                #copy_edge = edge
                #discarted_node = edge.n2 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n1
                #selected = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2
                #selected = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2
                sortedDistList.remove(edge)


            #if sol.of != sol.getEvalComplete(): #just to debugging purposes!!!!
            #    print("Erorrrrrr",sol.of,sol.getEvalComplete())
            return sol


    def ConstructiveHeuristic_without_last_move_safety(self, sol, safety = 0):
            sol.of = 0
            sortedDistList = self.copySortedEdgeList(self.instance.sortedDistances, sol) #Copy the edge list
            while (len(sortedDistList) != 0):
                #if not first_move:
                #    self.beta = 0.99
                #pos = 0 #Iterate the list in a greedy way
                #if isRandom:
                can = False
                while not can:
                    pos = self.getRandomPosition(len(sortedDistList), self.beta) #Induce a BR-behaviour
                    edge = sortedDistList[pos] #select and delect the edge of the position "pos" of the list
                    can = True if not edge.n1.used and not edge.n2.used else False
                    if not can:
                        del sortedDistList[sortedDistList.index(edge)]

                rand = random.random()
                #nodeMinCap = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2 # select the node with minimum capacity
                nodeMinCap = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2

                if (sol.capacity - nodeMinCap.capacity) >= (self.instance.minCapacity - safety*self.instance.minCapacity): #I can delete the node because I have not violate the demand
                    nodeMinCap.used = True
                    del sol.selected[sol.selected.index(nodeMinCap.id)] #delete the node with minimum capacity of the solution
                    sol.nonUsedNodes.append(nodeMinCap.id)
                    sol.capacity -= nodeMinCap.capacity #substract the capacity of the node of the solution.
                    #for e in nodeMinCap.associatedEdges: #update the edges involves in the deleted node
                    #    if e in sortedDistList and e!=edge:
                    #        del sortedDistList[sortedDistList.index(e)] #borro los edges asociados
                else: #I can not delete the node due to capacity constraints.

                    can = False
                    k = 0
                    while not can:
                        minEdge = sortedDistList[k]
                        if not minEdge.n1.used and not minEdge.n2.used:
                            can = True
                        k += 1

                    sol.of = minEdge.distance
                    sol.vMin1 = minEdge.n1.id
                    sol.vMin2 = minEdge.n2.id

                    # min_arista = min(non_discarted.associatedEdges)
                    # min([i.distance for i in non_discarted.associatedEdges if i.id in sol.selected.id and i.id != sol.vMin1])
                    # if minEdge <

                    break
                #copy_edge = edge
                #discarted_node = edge.n2 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n1
                #selected = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2
                #selected = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2
                sortedDistList.remove(edge)


            #if sol.of != sol.getEvalComplete(): #just to debugging purposes!!!!
            #    print("Erorrrrrr",sol.of,sol.getEvalComplete())
            return sol

    def ConstructiveHeuristic_without_last_move_simulation(self, sol, simhulation = simheuristic(20, 0.1)):
            sol.of = 0
            #simhulation = simheuristic(20,)
            sortedDistList = self.copySortedEdgeList(self.instance.sortedDistances, sol) #Copy the edge list
            while (len(sortedDistList) != 0):
                #if not first_move:
                #    self.beta = 0.99
                #pos = 0 #Iterate the list in a greedy way
                #if isRandom:
                can = False
                while not can:
                    pos = self.getRandomPosition(len(sortedDistList), self.beta) #Induce a BR-behaviour
                    edge = sortedDistList[pos] #select and delect the edge of the position "pos" of the list
                    can = True if not edge.n1.used and not edge.n2.used else False
                    if not can:
                        del sortedDistList[sortedDistList.index(edge)]

                rand = random.random()
                #nodeMinCap = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2 # select the node with minimum capacity
                nodeMinCap = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2
                simhulation.fast_simulation(sol)
                if (sol.capacity - nodeMinCap.capacity) >= (self.instance.minCapacity): #I can delete the node because I have not violate the demand
                    nodeMinCap.used = True
                    del sol.selected[sol.selected.index(nodeMinCap.id)] #delete the node with minimum capacity of the solution
                    sol.nonUsedNodes.append(nodeMinCap.id)
                    sol.capacity -= nodeMinCap.capacity #substract the capacity of the node of the solution.
                    #for e in nodeMinCap.associatedEdges: #update the edges involves in the deleted node
                    #    if e in sortedDistList and e!=edge:
                    #        del sortedDistList[sortedDistList.index(e)] #borro los edges asociados
                else: #I can not delete the node due to capacity constraints.

                    can = False
                    k = 0
                    while not can:
                        minEdge = sortedDistList[k]
                        if not minEdge.n1.used and not minEdge.n2.used:
                            can = True
                        k += 1

                    sol.of = minEdge.distance
                    sol.vMin1 = minEdge.n1.id
                    sol.vMin2 = minEdge.n2.id

                    # min_arista = min(non_discarted.associatedEdges)
                    # min([i.distance for i in non_discarted.associatedEdges if i.id in sol.selected.id and i.id != sol.vMin1])
                    # if minEdge <

                    break
                #copy_edge = edge
                #discarted_node = edge.n2 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n1
                #selected = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2
                #selected = edge.n1 if edge.n1.capacity < edge.n2.capacity and rand < 0.7 else edge.n2
                sortedDistList.remove(edge)


            #if sol.of != sol.getEvalComplete(): #just to debugging purposes!!!!
            #    print("Erorrrrrr",sol.of,sol.getEvalComplete())
            return sol

    '''
    Add a p% of new nodes (not used) in the solution 
    and apply the constructive heuristic
    '''
    def pertubation(self,baseSol,p):
        newSol = baseSol.copySol() #Copy the baseSol
        self.addNodesToSol(newSol, p) #Include new nodes in the solution
        newSol = self.ConstructiveHeuristic(newSol,False) #Apply the constructive Heuristic
        return newSol



    '''
    Select the link which is causing the bottneckle 
    and tries to substitute by a new one
    '''
    def localSearch(self,sol):

        improvement = True
        while improvement:
            # 1. Select the edge with minimum distance between the nodes of the Solution (bottleneck) and take the
            # node with lower capacity
            nodeMinCap = self.instance.nodes[sol.vMin1] if self.instance.nodes[sol.vMin1].capacity < self.instance.nodes[sol.vMin2].capacity else self.instance.nodes[sol.vMin2]
            minDist = sys.maxsize

            #2. Compute the minimum capacity an minimum distance required to accept a new node in the solution
            mincapacity = self.instance.minCapacity - (sol.capacity - nodeMinCap.capacity)
            CurrentMinDistance = sol.of

            #3. Look for a new node in order to be included in the solution
            for nOutSol in sol.nonUsedNodes:
                ValidNode = nOutSol
                for nInSol in sol.selected:
                    distanceBetweenNodes = self.instance.distance[nInSol][nOutSol]
                    if CurrentMinDistance >= distanceBetweenNodes or self.instance.nodes[nOutSol].capacity < mincapacity:
                        ValidNode = -1 #not valid node to be included in the solution
                        break
                if ValidNode != -1:
                    improvement = True
                    break
                else:
                    improvement = False

            #4 If the solution can be improved I carryOut a swap between nodes
            if improvement:
                #4.1 delete the node of the bottleneck edge with minimum capacity
                del sol.selected[sol.selected.index(nodeMinCap.id)]  # borro el nodo
                sol.nonUsedNodes.append(nodeMinCap.id) #añado el nodo a la lista de nodos no usados
                sol.capacity -= self.instance.nodes[nodeMinCap.id].capacity

                #4.2 Include the new Node
                del sol.nonUsedNodes[sol.nonUsedNodes.index(nOutSol)]  # delete the node of the nonUsedNodes List
                sol.selected.append(nOutSol)  # add the nodes to the Solution
                sol.capacity += self.instance.nodes[nOutSol].capacity  # Increase capacity to he solution

                #4.3 Compute the new Objective Function value (Minimum distance between nodes)
                of = self.instance.distance[sol.selected[0]][sol.selected[1]]

                sol.of = of
                sol.vMin1 = sol.selected[0]
                sol.vMin2 = sol.selected[1]

                for k, i in enumerate(sol.selected[:-1]): #
                    for j in sol.selected[(k+1):]: #[(i+1):]
                        if of > self.instance.distance[i][j]:
                            of = self.instance.distance[i][j]
                            sol.of = self.instance.distance[i][j]
                            sol.vMin1 = i
                            sol.vMin2 = j


        return sol

    def localSearch_safety(self,sol,safety = 0):

        improvement = True
        while improvement:
            # 1. Select the edge with minimum distance between the nodes of the Solution (bottleneck) and take the
            # node with lower capacity
            nodeMinCap = self.instance.nodes[sol.vMin1] if self.instance.nodes[sol.vMin1].capacity < self.instance.nodes[sol.vMin2].capacity else self.instance.nodes[sol.vMin2]
            minDist = sys.maxsize

            #2. Compute the minimum capacity an minimum distance required to accept a new node in the solution
            mincapacity = (self.instance.minCapacity - safety*self.instance.minCapacity) - (sol.capacity - nodeMinCap.capacity)
            CurrentMinDistance = sol.of

            #3. Look for a new node in order to be included in the solution
            for nOutSol in sol.nonUsedNodes:
                ValidNode = nOutSol
                for nInSol in sol.selected:
                    distanceBetweenNodes = self.instance.distance[nInSol][nOutSol]
                    if CurrentMinDistance >= distanceBetweenNodes or self.instance.nodes[nOutSol].capacity < mincapacity:
                        ValidNode = -1 #not valid node to be included in the solution
                        break
                if ValidNode != -1:
                    improvement = True
                    break
                else:
                    improvement = False

            #4 If the solution can be improved I carryOut a swap between nodes
            if improvement:
                #4.1 delete the node of the bottleneck edge with minimum capacity
                del sol.selected[sol.selected.index(nodeMinCap.id)]  # borro el nodo
                sol.nonUsedNodes.append(nodeMinCap.id) #añado el nodo a la lista de nodos no usados
                sol.capacity -= self.instance.nodes[nodeMinCap.id].capacity

                #4.2 Include the new Node
                del sol.nonUsedNodes[sol.nonUsedNodes.index(nOutSol)]  # delete the node of the nonUsedNodes List
                sol.selected.append(nOutSol)  # add the nodes to the Solution
                sol.capacity += self.instance.nodes[nOutSol].capacity  # Increase capacity to he solution

                #4.3 Compute the new Objective Function value (Minimum distance between nodes)
                of = self.instance.distance[sol.selected[0]][sol.selected[1]]

                sol.of = of
                sol.vMin1 = sol.selected[0]
                sol.vMin2 = sol.selected[1]

                for k, i in enumerate(sol.selected[:-1]): #
                    for j in sol.selected[(k+1):]: #[(i+1):]
                        if of > self.instance.distance[i][j]:
                            of = self.instance.distance[i][j]
                            sol.of = self.instance.distance[i][j]
                            sol.vMin1 = i
                            sol.vMin2 = j


        return sol



    '''
    Open a p% of nodes not used in the solution 
    :parameter Sol: Solution
    :parameter p: percentage of nodes to be open
    '''
    def addNodesToSol(self,sol,p):
        NumberOfNodesToOpen = math.ceil(len(sol.nonUsedNodes) * p / 100)  # Number of nodes to delete
        if NumberOfNodesToOpen != 0:
            for _ in range(0, NumberOfNodesToOpen):
                n = random.choice(sol.nonUsedNodes)
                del sol.nonUsedNodes[sol.nonUsedNodes.index(n)] #delete the node of the nonUsedNodes List
                sol.selected.append(n)  #add the nodes to the Solution
                sol.capacity += self.instance.nodes[n].capacity #Increase capacity to he solution
                for e in self.instance.nodes[n].associatedEdges:  # update the edges involves in the deleted node
                    if not e.n2.id in sol.nonUsedNodes and not e.n1.id in sol.nonUsedNodes:
                        e.isInSol = True




    def copySortedEdgeList(self, edgeList,sol):
        newList = []
        for e in edgeList:
            if not e.n2.id in sol.nonUsedNodes and not e.n1.id in sol.nonUsedNodes:
                newList.append(e)
        return newList



    '''
    Obtain a random number between 0 and the maximum size of a list using a Geometric distribution behavior.
    parameters:
    Size: Size of a list
    beta: Beta parameter of the Geometric distribution
    '''
    def getRandomPosition(self,size,beta):
        index = int(math.log(random.random()) / math.log(1 - beta))
        index = index % size
        return index

