import numpy as np
from objects import Edge

class Instance:

    def __init__(self,path):
        self.name = ""  # nombre de la instancia
        self.n = 0  # nodos
        self.b = 0  # capacidad mínima requerida
        self.capacity = []  # vector de capacidades
        self.distance = None  # matriz de distancia
        self.sortedDistances = []  # lista ordenada de Edge
        self.readInstance(path)


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
                    self.b = int(line)
                elif i == 3: #capacidades nodos
                    l = line.rstrip('\t\n ')
                    self.capacity = [float(x) for x in l.split('\t')]
                else: #matriz distancia
                    l = line.rstrip('\t\n ')
                    d= [float(x) for x in l.split('\t')]
                    for z in range(0,self.n):
                            if d[z] != 0:
                                self.distance[fila,z] = d[z]
                                self.sortedDistances.append(Edge(fila,z,d[z]))
                    fila+=1
                i += 1
        self.sortedDistances.sort(key=lambda x: x.distance, reverse=True) #ordena distancia de mayor a menos