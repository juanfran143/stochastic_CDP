#Class Candidate
class Candidate:

    def __init__(self, v, closestV, cost):
        self.v = v
        self.closestV = closestV
        self.cost = cost

class Candidate_capacity:

    def __init__(self, v, closestV, dist_min, cost):
        self.v = v
        self.closestV = closestV
        self.dist_min = dist_min

        self.cost = cost




#Class Edge
class Edge:

    def __init__(self,v1, v2, distance):
        self.v1 = v1
        self.v2 = v2
        self.distance = distance





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