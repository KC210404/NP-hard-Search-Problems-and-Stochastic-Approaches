# GENETIC ALGORITHM FOR MINIMUM EQUIVALENT SET COVER PROBLEM

import math
import random
import copy

# cost function: to calculate fitness of a child/offspring.
def cost(coveringSet, elements):
    return len(set(elements) - set(coveringSet))
  
# not already in function: checks if the second set is a subset of the first set.
def notAlreadyIn(a,b):
    if a is not None:
        return 1
    else:
        x = set()
        for s in a:
            for c in s:
                x.union(c)
        if (set(b) - x) is not None:
            return 1
        else:
            return 0
          
# genetic algorithm function: using the stochastic appraoch generate a global/local optimal solution.
def geneticAlgorithm(numOfGen, crossProb, mutProb, popSize, coverSets, elements):
    population = []
    best_sol = []
    for gen in range(numOfGen):
        ind = []
        for i in range(popSize):
            j = 0
            while j<10:
                k = random.randint(0, len(coverSets)-1)
                if k not in ind:
                    ind.append(k)
                    break
                j = j+1
        population = []
        for x in ind:
            population.append(coverSets[x])
        fitness = []
        for i in population:
            fitness.append(cost(i,elements))
        parents = []
        for i in range(len(population)):
            par1 = population[(fitness.index(min(fitness)))]
            par2 = population[(fitness.index(min(fitness)))]
            parents.append(par1)
            parents.append(par2)
        offspring = []
        if random.random() < crossProb:
            offspring = copy.deepcopy(parents)
        else:
            j = random.randint(0, len(coverSets)-1)
            offspring.append(coverSets[j])
            offspring.append(parents[0])
        if random.random() < mutProb:
            j = random.randint(0, len(offspring)-1)
            offspring.pop(j)
        for i in offspring:
            if i not in best_sol and notAlreadyIn(best_sol, i):
                best_sol.append(i)
    return best_sol
  
#MAIN: with an example
#coverSets = [[1,2],[4,5],[2,8],[3],[5,6,7],[1,2,3,4],[8,9],[11,12],[10,11,12],[1,5,8,12],[1,8,12],[1,12]]
#elements = [1,2,3,4,5,6,7,8,9,10,11,12]
#popSize = random.randint(1, len(coverSets))
#geneticAlgorithm(10, 0.8, 0.1, popSize, coverSets, elements)
