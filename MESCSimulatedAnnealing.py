# SIMULATED ANNEALING ALGORITHM FOR MINIMUM EQUIVALENT COVERING SET PROBLEM
import math
import random
import copy
import numpy as np

# Cost function: estimates the cost of each combination of covering sets from the initial covering set.
def cost(coveringSets, elements):
    covered = set()
    for s in coveringSets:
        covered |= set(s)
    return len(set(elements) - covered)
  
# num function: estimates the number of elements in all the sets of a given combination of the initial covering set
def num(coveringSets):
    i = 0
    for s in coveringSets:
        i += len(s)
    return i
  
# equality function: Checks the equality for the provided combinations of the initial covering set
def equality(a,b):
    if len(a)!=len(b):
        return 0
    else:
        for i in range(len(a)):
            if len(a[i])!=len(b[i]):
                return 0
            for j in range(len(a[i])):
                if a[i][j]!=b[i][j]:
                    return 0
        return 1
      
# simulated annealing function: Using the simulated annealing stochastic approach solves for the optimal solution and returns it
def simulatedAnnealing(coveringSet, elements, iniTemp, coolingStep, maxIters):
    currSol = copy.deepcopy(coveringSet)
    currCost = cost(currSol, elements)
    bestSol = copy.deepcopy(coveringSet)
    bestCost = currCost
    temp = iniTemp
    altSol = []
    for i in range(maxIters):
        if temp == 0:
            break
        altSol = copy.deepcopy(currSol)
        if altSol:
            a = random.randint(0, len(altSol) - 1)
            altSol.pop(a)
        else:
            a = random.randint(0, len(coveringSet) - 1)
            altSol = copy.deepcopy(coveringSet)
            altSol.pop(a)
        altCost = cost(altSol, elements)
        delta = altCost - currCost
        if delta < 0 or math.exp(-delta / temp) > random.random():
            currSol = altSol
            currCost = altCost
        if altCost <= bestCost and len(altSol) <= len(bestSol):
            bestSol = altSol
            bestCost = altCost
        temp = temp*coolingStep
    return bestSol, bestCost
  
#variation of the simulated annealing function for automation of aquisition of a training data set. The goal is to be supplied to the function, so that it returns the iteration number where the goal is our best solution.
#def simulatedAnnealing(coveringSet, elements, iniTemp, coolingStep, goal):
#    currSol = copy.deepcopy(coveringSet)
#    currCost = cost(currSol, elements)
#    bestSol = copy.deepcopy(coveringSet)
#    bestCost = currCost
#    temp = iniTemp
#    altSol = []
#    Iter = 0
#    while Iter<5000:
#        Iter += 1
#        if temp == 0:
#            break
#        altSol = copy.deepcopy(currSol)
#        if altSol:
#            a = random.randint(0, len(altSol) - 1)
#            altSol.pop(a)
#        else:
#            a = random.randint(0, len(coveringSet) - 1)
#            altSol = copy.deepcopy(coveringSet)
#            altSol.pop(a)
#        altCost = cost(altSol, elements)
#        delta = altCost - currCost
#        if delta < 0 or math.exp(-delta / temp) > random.random():
#            currSol = altSol
#            currCost = altCost
#        if altCost <= bestCost and num(altSol) <= num(bestSol):
#            bestSol = altSol
#            bestCost = altCost
#            if equality(bestSol,goal):
#                return Iter
#        temp = temp*coolingStep
# ML function for tuning parameters of simulated annealing

def MLSimAnnealing(coveringSet, elements):
    file = open("../DataSetSim.txt","r")
    x1 = len(coveringSet), z1 = len(elements), a1 = num(coveringSet)
    x = [], y = [], z = [], a = []
    for s in file:
    	s = eval(s)
        if s[5]:
            x.append(len(s[0]))
            y.append(s[3])
            z.append(len(s[1]))
            a.append(num(s[0]))
    poly1 = np.polyfit(x,y,3)
    poly2 = np.polyfit(z,y,3)
    poly3 = np.polyfit(a,y,3)
    i = 3
    eq1 = 0, eq2 = 0, eq3 = 0
    while i>=0:
        y1 += poly1*(x1**(3-i))
        y2 += poly2*(z1**(3-i))
        y3 += poly3*(a1**(3-i))
        i -= 1
    yres = (x1*y1+z1*y1+a1*y1)/(x1+a1+z1)
    maxIter = (int)yres
    iniTemp = 1000.0
    coolingStep = 10**(-9/maxIter)
    return iniTemp, coolingStep, maxIters
  
# Main: for automated aquisition of training data set for ML
#iniTemp = 1000.0
#coolingStep = 0.95
#file1 = open("../TrainingSetSim.txt","r")
#file2 = open("../DataSetSim.txt","w")
#for s in file1:
#    print(1)
#    s = eval(s)
#    i = 0 
#    x= []
#    while i < 100:
#        n = simulatedAnnealing(s[1],s[0],iniTemp,coolingStep,s[2])
#        if n is not None:
#            x.append(n)
#        i += 1
#    maxIter = getMaxIter(x, accuracy=10)
#    file2.write([s[1], s[0], coolingStep, maxIter, s[2], True])
#file1.close()
#file2.close()
#print("finished")

#MAIN: using ML function too:-
#coveringSet = [...]
#elements = [...]
#iniTemp, coolingStep, maxIters = MLSimAnnealing(coveringSet, elements)
#best_soln, best_cost = simulatedAnnealing(coveringSet, elements, iniTemp, coolingStep, maxIters)
#print(best_soln)
