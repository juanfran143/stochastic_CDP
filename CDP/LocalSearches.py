from Solution import *
import random

# Tabu Search LS
def tabuSearch(initSol, cl, maxIter, heur):
    sol = initSol.copySol()
    noImprovement = 0
    while noImprovement < maxIter:
        n = sol.selected[0] #select the oldest element
        sol.drop(n) #drop the oldest node of the solucion
        heur.recalculateCL(sol, cl, n)
        sol = heur.partialReconstruction(sol, cl)    #reconstruct the solution without using the oldest node
        sol.reevaluateSol()  # update of
        heur.insertNodeToCL(cl, sol, n)
        if sol.of > initSol.of:
            initSol = sol.copySol()
            noImprovement = 0
        noImprovement += 1
    return initSol



def tabuSearch_capacity(initSol, cl, maxIter, heur):
    sol = initSol.copySol()
    noImprovement = 0
    while noImprovement < maxIter:
        n = sol.selected[0] #select the oldest element
        sol.drop(n) #drop the oldest node of the solucion
        heur.recalculateCL_capacity(sol, cl, n)
        sol = heur.partialReconstruction_capacity(sol, cl)  ####  #reconstruct the solution without using the oldest node
        sol.reevaluateSol()  # update of
        heur.insertNodeToCL_capacity(cl, sol, n)
        if sol.of > initSol.of:
            initSol = sol.copySol()
            noImprovement = 0
        noImprovement+=1
    return initSol


def tabuSearch_capacity_simulation(initSol, cl, maxIter, heur, simulation, delta):
    sol = initSol.copySol()
    noImprovement = 0
    while noImprovement < maxIter:
        n = sol.selected[0] #select the oldest element
        sol.drop(n) #drop the oldest node of the solucion
        heur.recalculateCL_capacity(sol, cl, n)
        sol = heur.partialReconstruction_capacity_simulation(sol, cl, simulation, delta)  ####  #reconstruct the solution without using the oldest node
        sol.reevaluateSol()  # update of
        heur.insertNodeToCL_capacity(cl, sol, n)
        if sol.of > initSol.of:
            initSol = sol.copySol()
            noImprovement = 0
        noImprovement+=1
    return initSol


def Sx_y(lst1, lst2):
    return [value for value in lst1 if value not in lst2]

def create_dic(sol, instance):
    dic_sol = dict([(i, (-1, -1)) for k, i in enumerate(sol.selected)])
    of = -1
    for k, i in enumerate(sol.selected):
        for j in sol.selected:
            if j == i:
                continue  # sol.selected[:k] + sol.selected[k+1:]
            if dic_sol[i][1] > instance.distance[j][i] or dic_sol[i][1] == -1:
                dic_sol[i] = (j, instance.distance[j][i])
            if of == -1 or of > instance.distance[j][i]:
                of = instance.distance[j][i]
    sol.of = of
    return dic_sol

def build_RL(dic_sol, sol, alpha):
    RL = []
    for i in dic_sol.keys():
        if sol.of >= alpha * dic_sol[i][1]:
            RL.append((i, dic_sol[i]))  # No metemos la distancia porque no queremos hacer el sort
    RL = dict(RL)
    return RL

def first_f(instance, sol, i, j):
    first = True
    cost = instance.distance[sol.selected[0]][i] + 1  # Le sumo uno para ahorrarme la comprobación de si es -1
    for k in sol.selected:
        if cost > instance.distance[i][k]:
            cost = instance.distance[i][k]
            if cost < j[1][1]:
                first = False
                break
    return first

def do(S_all_sol, instance, sol, j, capacity_old, RL):
    improve = False
    for i in S_all_sol:
        if sol.capacity - capacity_old + instance.capacity[i] < sol.instance.b:
            continue

        first = first_f(instance, sol, i, j)

        if first:
            improve = True
            sol.selected.append(i)
            S_all_sol.remove(i)
            sol.capacity = sol.capacity - capacity_old + instance.capacity[i]
            break
    return improve

def RL_tabuSearch(initSol: Solution, maxIter, alpha, all_nodes_t):
    best_sol = initSol.copySol()
    sol = initSol.copySol()
    instance = initSol.instance
    #all_nodes = [i for i in range(0, len(instance.capacity))]
    all_nodes = all_nodes_t.copy()
    first_time = True
    for _ in range(maxIter):
        of_before = sol.of
        S_all_sol = Sx_y(all_nodes, sol.selected)
        dic_sol = create_dic(sol, instance)
        if not first_time and sol.of == of_before:
            break
        else:
            first_time

        if best_sol.of < sol.of:
            best_sol = sol.copySol()

        RL = build_RL(dic_sol, sol, alpha)

        remove = []
        for j in RL.items():

            if j[1][0] in remove or j[0] in remove:
                continue

            get = j[0] if instance.capacity[j[0]] < instance.capacity[j[1][0]] else j[1][0]
            capacity_old = instance.capacity[get]
            remove.append(get)
            sol.selected.remove(get)

            improve = do(S_all_sol, instance, sol, j, capacity_old, RL)

            if not improve:
                sol.selected.append(get)
                remove.pop()

    return best_sol





