import numpy as np

#Class Edge
class Edge:
    def __init__(self,n1, n2, distance):
        self.n1 = n1
        self.n2 = n2
        self.distance = distance
        self.cost = 0
        self.isInSol = True
        self.used = False


#Class Node
class Node:
    def __init__(self,id,capacity):
        self.id = id
        self.capacity = capacity
        self.associatedEdges = [] #List of the Edges associated to this node
        self.used = False


#class Test
#contain the parameters of the execution
class Test:
    def __init__(self, instName, seed, time, beta1, beta2, maxIter, delta, short_simulation, long_simulation, var, deterministic, not_penalization_cost, weight, inversa):
        self.instName = instName #Instance Name
        self.Maxtime = int(time) #max Execution Time
        self.betaBR = float(beta1) #beta BR 1
        self.betaLS = float(beta2) #beta BR 2
        self.seed = int(seed) #seed
        self.maxIter = int(maxIter) #seed
        self.delta = float(delta)
        self.short_simulation = int(short_simulation)
        self.long_simulation = int(long_simulation)
        self.var = float(var)
        self.deterministic = True if deterministic == "True" else False
        self.not_penalization_cost = True if not_penalization_cost == "True" else False
        self.weight = float(weight)
        self.inversa = float(inversa)




class Instance:
    def __init__(self):
        self.name = ""  # nombre de la instancia
        self.n = 0  # nodos
        self.minCapacity = 0  # capacidad mínima requerida
        self.distance = None  # matriz de distancia
        self.sortedDistances = []  # lista ordenada de Edge
        self.nodes = []


    def readInstance(self, s):
        with open(s) as instance:
            i = 1
            fila = 0
            for line in instance:
                if line == "\n":
                    continue
                if i == 1: #nodos
                    self.n = int(line)
                    self.distance = np.zeros((self.n, self.n))
                elif i == 2: #capacidad
                    self.minCapacity = int(line)
                elif i == 3: #capacidades nodos
                    l = line.rstrip('\t\n ').split("\t")
                    for i in range(0,len(l)):
                        self.nodes.append(Node(i,float(l[i])))
                else: #matriz distancia
                    l = line.rstrip('\t\n ')
                    d = [float(x) for x in l.split('\t')]
                    for z in range(fila, self.n):
                        if z != fila:
                            self.distance[fila, z] = d[z]
                            self.distance[z, fila] = d[z]
                            edge = Edge(self.nodes[fila], self.nodes[z], d[z])
                            self.sortedDistances.append(edge)
                            self.nodes[fila].associatedEdges.append(edge)
                            self.nodes[z].associatedEdges.append(edge)
                        """
                            if d[z] != 0:
                                self.distance[fila,z] = d[z]
                                self.distance[z, fila] = d[z]
                                edge = Edge(self.nodes[fila],self.nodes[z],d[z])
                                self.sortedDistances.append(edge)
                                self.nodes[fila].associatedEdges.append(edge)
                                self.nodes[z].associatedEdges.append(edge)
                        """
                    fila+=1
                i += 1
        self.sortedDistances.sort(key=lambda x: x.distance) #ordena distancia de menor a mayor distancia