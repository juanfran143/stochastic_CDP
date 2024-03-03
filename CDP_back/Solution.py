import sys

# Class Solution
class Solution:

    def __init__(self, instance):
        self.instance = instance
        self.selected = [n for n in range(0, instance.n)]
        self.vMin1 = -1
        self.vMin2 = -1
        self.of = 0
        self.capacity = sum([instance.nodes[n].capacity for n in range(0, instance.n)])
        self.time = 0
        self.nonUsedNodes = []

        self.reliability = {"1": 0, "2": 0}
        self.total_stochastic_capacity = {"1": 0, "2": 0}
        self.stochastic_of = {"1": 0, "2": 0}
        self.mean_stochastic_of = {"1": 0, "2": 0}

    def copySol(self):
        newSol = Solution(self.instance)
        newSol.selected.clear()
        newSol.vMin1 = self.vMin1
        newSol.vMin2 = self.vMin2
        newSol.of = self.of
        newSol.capacity = self.capacity
        newSol.time = self.time
        for i in self.selected:
            newSol.selected.append(i)
        for i in self.nonUsedNodes:
                newSol.nonUsedNodes.append(i)
        return newSol


    def getEvalComplete(self):
        distance = 99999
        for s1 in self.selected:
            for s2 in self.selected:
                if s1 == s2:
                    continue
                d = self.instance.distance[s1][s2]
                if d < distance:
                    distance = d
        return distance

    def distanceTo(self, v):
        minDist = self.instance.sortedDistances[0].distance * 10
        vMin = -1
        for s in self.selected:
            d = self.instance.distance[s][v]
            if d < minDist:
                minDist = d
                vMin = s
        return vMin, minDist



