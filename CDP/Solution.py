# Class Solution
class Solution:

    def __init__(self, instance):
        self.instance = instance
        self.selected = []
        self.vMin1 = -1
        self.vMin2 = -1
        self.of = instance.sortedDistances[0].distance*10
        self.capacity = 0
        self.time = 0

        self.reliability = {"1": 0, "2": 0}
        self.total_stochastic_capacity = {"1": 0, "2": 0}
        self.stochastic_of = {"1": 0, "2": 0}
        self.mean_stochastic_of = {"1": 0, "2": 0}



    def copySol(self):
        newSol = Solution(self.instance)
        newSol.vMin1 = self.vMin1
        newSol.vMin2 = self.vMin2
        newSol.of = self.of
        newSol.capacity = self.capacity
        newSol.time = self.time

        for i in range(2):
            newSol.reliability[str(i+1)] = self.reliability[str(i+1)]
            newSol.total_stochastic_capacity[str(i+1)] = self.total_stochastic_capacity[str(i+1)]
            newSol.stochastic_of[str(i+1)] = self.stochastic_of[str(i+1)]
            newSol.mean_stochastic_of[str(i+1)] = self.mean_stochastic_of[str(i+1)]

        for i in self.selected:
            newSol.selected.append(i)
        return newSol


    def add(self,v):
        self.selected.append(v)
        self.capacity += self.instance.capacity[v]

    def drop(self,v):
        index = self.selected.index(v)
        del self.selected[index]
        self.capacity -= self.instance.capacity[v]
    


    def distanceTo(self,v):
        minDist = self.instance.sortedDistances[0].distance * 10
        vMin = -1
        for s in self.selected:
            d = self.instance.distance[s][v]
            if d < minDist:
                minDist = d
                vMin = s
        return vMin, minDist


    def isFeasible(self):
        return self.capacity >= self.instance.b


    def updateOF(self,vMin1,vMin2, of):
        self.of = of
        self.vMin1 = vMin1
        self.vMin2 = vMin2



    def getEvalComplete(self):
        self.of = self.instance.sortedDistances[0].distance * 10
        for s1 in self.selected:
            for s2 in self.selected:
                if s1 == s2:
                    continue
                d = self.instance.distance[s1][s2]
                if d < self.of:
                    self.of = d
        return self.of



    def reevaluateSol(self):
        self.of = self.instance.sortedDistances[0].distance * 10
        for s1 in self.selected:
            for s2 in self.selected:
                if s1 == s2:
                    continue
                d = self.instance.distance[s1][s2]
                if d < self.of:
                    self.of = d
                    self.vMin1 = s1
                    self.vMin2 = s2
