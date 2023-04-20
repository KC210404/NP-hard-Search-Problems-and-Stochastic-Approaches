# NP-hard-Search-Problems-and-Stochastic-Approaches-REPORT

1	Introduction:

1.1 NP-hard Problems:

NP-hard problems are decision problems that are at least as hard as the hardest problems in the complexity class NP (Non-Polynomial Deterministic). NP-hard search problems are optimization problems that are at least as hard as the hardest problems in the complexity class NP. An optimization problem is a problem that requires finding the best solution among a set of possible solutions.
An example of an NP-hard search problem is the traveling salesman problem, which asks for the shortest possible route that visits a set of cities exactly once and returns to the starting city. If a polynomial-time algorithm can be found for any NP-hard search problem, then it would imply that all NP-hard decision problems can be solved in polynomial time. However, as with NP-hard decision problems, no such algorithm has been found to date. Therefore, researchers have developed approximation algorithms and heuristic methods to find near-optimal solutions to these problems in practice.

1.2 Stochastic Search Approaches/ Algorithmic statergy:

Stochastic search approaches are a class of optimization algorithms that use randomness to explore and search for optimal solutions in a search space. Unlike deterministic methods, such as gradient descent, which follow a fixed path to reach an optimum, stochastic search algorithms explore the search space through random sampling of solutions.
The most popular and highly used stochasic search algorithms:
1. Simulated Annealing: a probabilistic technique that uses an analogy with the annealing process in metallurgy to find a near-optimal solution. It starts with a high-temperature state and gradually reduces the temperature to allow for the exploration of a larger search space.
2. Genetic Algorithms: a search algorithm inspired by the process of natural selection. It maintains a population of candidate solutions, applies mutation and crossover operators to generate new solutions, and selects the best ones for the next generation.

1.3 Choosing Stochastic stratergies for solving NP-hard Search Problems:

Stochastic algorithms offer a promising approach for solving NP-hard search problems, which are notoriously difficult to solve using traditional methods. These algorithms leverage randomness to explore the search space and find near-optimal solutions, even when the problem has a large and complex search space with many local optima. Stochastic algorithms offer flexibility, robustness, and scalability that are essential for solving NP-hard search problems.
Stochastic algorithms are flexible and adaptable, making them suitable for solving a variety of optimization problems with complex constraints and objectives. They are particularly useful for NP-hard search problems, which can have many constraints and objectives that are difficult to model and optimize using traditional methods. Stochastic algorithms are also more robust than deterministic algorithms when dealing with noisy or incomplete data, which is common in NP-hard search problems. Furthermore, many stochastic algorithms can be parallelized, which allows them to handle large-scale optimization problems with many variables and constraints. This scalability is essential for NP-hard search problems, which can have a large number of variables and constraints that are difficult to optimize using traditional methods.
Stochastic algorithms can offer significant time complexity improvements when solving NP-hard problems compared to deterministic algorithms. This is because stochastic algorithms can explore the search space more efficiently by leveraging randomness and avoiding getting stuck in local optima. In contrast, deterministic algorithms may require exponential time to explore the entire search space, making them infeasible for many NP-hard problems. Stochastic algorithms can also be parallelized, which further reduces the computation time for large-scale optimization problems. However, it is important to note that stochastic algorithms do not guarantee an optimal solution and may require multiple runs with different initial conditions to ensure convergence to a good solution. Therefore, the trade-off between computation time and solution quality should be carefully considered when using stochastic algorithms for NP-hard problems.

1.4 Minimum Equivalent Cover Set Problem (MECS):

The Minimum Equivalent Cover Set (MECS) problem is a type of NP-hard search problem. Given a set of subsets of a finite universe, the goal of the MECS problem is to find the minimum set of subsets that can cover the universe, such that any other subset can be represented as the union of a subset of the chosen sets. In other words, the MECS problem seeks to find the smallest set of subsets that can represent all other subsets in the universe.
This problem is NP-hard, which means that it is computationally infeasible to solve it optimally in polynomial time using traditional methods. As such, stochastic algorithms, such as simulated annealing, genetic algorithms, and ant colony optimization, are often used to find approximate solutions to the MECS problem. These algorithms use randomness to explore the search space and find near-optimal solutions, and they can handle large and complex instances of the MECS problem.
Example: Suppose we have a universe of elements {1, 2, 3, 4, 5} and a set of subsets as follows:
S1 = {1, 2} S2 = {1, 3, 4} S3 = {2, 3} S4 = {4, 5}
Here, the smallest number of subsets required for getting the universe of elements when their are united (union operation) is – 3 as no union of binary combinations of sets is equivalent to the universal set.
The Answer might be any of the sets: {S1, S2, S4}, {S1, S3, S4}, {S2, S3, S4}, but to resolve the collision we could use another constraint- minimum number of elements repeating therefore, the answer would be: {S1, S3, S4}.

1.5 Applications of MECS:

1. Database Query Optimization: In database management systems, the MECS problem can be used to optimize queries by identifying the minimum set of indexes that can cover all possible query predicates.
2. Code Optimization: In compiler optimization, the MECS problem can be used to optimize code by identifying the minimum set of basic blocks that can cover all possible program paths.
3. Network Design: In network design, the MECS problem can be used to optimize the routing of data packets by identifying the minimum set of routers that can cover all possible paths between nodes in the network.
4. Feature Selection: In machine learning and data mining, the MECS problem can be used to identify the minimum set of features that can represent the entire dataset with a high degree of accuracy. This can help to reduce the dimensionality of the dataset and improve the performance of machine learning models.

1.6 Working principle of the Simulated Annealing Algorithm - Variation:

Simulated annealing is a stochastic optimization algorithm used to solve combinatorial optimization problems. The algorithm is inspired by the physical process of annealing, where a material is slowly cooled to reach a state of minimum energy. Similarly, simulated annealing starts with a random solution and gradually refines it by minimizing a cost function. The algorithm makes random changes to the current solution and evaluates the new solution. If the new solution has a lower cost, it is accepted as the new current solution. However, if the new solution has a higher cost, it may still be accepted with a certain probability, which decreases over time. This allows the algorithm to escape local optima and explore the search space more effectively.
In this document the Simualted Annealing Algorithm’s variation for solving MECS is such, it doesn’t accepts the solutions with higher costs. To do that, there was a trade-off implemented between checking the number of subsets in the best solution and in the alternate solution, so the complexity is little higher than expected, but the solution quality is improved.
The simulated annealing algorithm consists of the following steps (without the above variation):
1. Initialize the current solution randomly.
2. Set the initial temperature and cooling rate.
3. While the temperature is above a predefined stopping criterion, repeat the following steps:
a. Generate a new solution by making a small random change to the current solution. 
b. Compute the cost of the new solution. 
c. If the new solution has a lower cost, accept it as the new current solution.
d. If the new solution has a higher cost, accept it with a certain probability based on the current temperature and the cost difference between the new and current solutions.
e. Decrease the temperature according to the cooling rate.
4. Return the best solution found during the algorithm.

1.7 Working principle of the Genetic Algorithm:

Genetic algorithms are stochastic optimization algorithms based on the principles of natural selection and genetics. These algorithms are used to solve complex optimization problems by mimicking the process of evolution in nature. The algorithm maintains a population of potential solutions, and iteratively improves the population through a process of selection, crossover, and mutation.
The working principle of genetic algorithms can be summarized as follows:
1. Initialization: The algorithm starts by creating an initial population of potential solutions randomly.
2. Fitness Evaluation: Each solution in the population is evaluated based on a fitness function that measures how well the solution solves the problem.
3. Selection: Solutions with higher fitness are more likely to be selected for the next generation. Various selection techniques can be used, such as tournament selection or roulette wheel selection.
4. Crossover: Selected solutions are combined to create new solutions through a process of crossover or recombination. The crossover operator exchanges genetic information between the selected solutions to create new offspring with a mixture of traits from their parents.
5. Mutation: Occasionally, a random mutation is introduced to a solution to add diversity to the population. This helps to prevent premature convergence and allows the algorithm to explore new regions of the search space.
6. Replacement: The new offspring and mutated solutions are used to replace the least fit solutions in the population.
7. Termination: The algorithm continues to repeat the above steps until a stopping criterion is met, such as a maximum number of generations or a satisfactory level of fitness.

1.8 Parameter Tuning for the Stochastic algorithms and the Role of ML in it:

Parameter tuning is an important aspect of stochastic algorithms as it can significantly impact the performance and effectiveness of the algorithm. Stochastic algorithms have several parameters that can be adjusted, such as the population size, mutation rate, crossover rate, and cooling schedule in simulated annealing. The optimal values of these parameters depend on the problem being solved and the characteristics of the search space.
Traditionally, parameter tuning has been performed manually, where the parameters are adjusted by trial and error. However, this approach can be time-consuming and may not lead to the best solution. Machine learning (ML) techniques can be used to automate the parameter tuning process and improve the efficiency and effectiveness of stochastic algorithms.
ML-based parameter tuning involves using a machine learning model to predict the optimal values of the parameters based on the problem and search space characteristics. The model is trained on a set of parameter configurations and their corresponding performance metrics, such as the fitness of the solution or the convergence time. The trained model can then be used to predict the optimal values of the parameters for new problem instances.

1.9 Time Complexity of MECS:

Since, MECS is an NP-hard search Problem, therefore it can’t be solved in polynomial time complexity, and its worst case time complexity is exponential. 
Calculations:
In worst-case, all the combinations of subsets, associated with a given universe are to be explored for providing us with the optimal solution. Therefore, for an initial Covering set of n elements associated with a universe the number of combinations that are formed are:
Number of Combinations of 1 subset:  nC1 
Number of Combinations of 2 subsets: nC2
.....
Number of Combinations of k < n subsets: nCk
Number of Combinations of n subsets: nCn
Therefore, total number of combinations = nC1 + nC2  +.......+ nCk +.......+ nCn = 2n – 1
That is worst-case time complexity for an algorithm which performs union between all the possible combinations of subsets in the initial covering set and explores for the optimal solution is: 
O(k*2n) where k is the number of elements in the all the subsets of the initial covering set.
That is for a MECS problem with 13 elements/ vertices in the universe, and 20 elements in all the subsets of the initial covering set, the expected number of iterations are – 20* 213 = 1,63,840 iterations, which is enormous and as the number of elements in the universe and the subsets of the initial covering set increases, the iterations increase exponentially and the required computational power is also very high.


2	Literature Survey:

1. "A Genetic Algorithm for the Minimum Equivalent Cover Set Problem" by D. Whitley and L. J. Barlow: https://doi.org/10.1109/ICEC.1996.542238

2. "A hybrid genetic algorithm for the minimum equivalent cover set problem" by C. P. Lim and P. B. Vidyasagar: https://doi.org/10.1016/j.cie.2008.03.025

3. "A simulated annealing algorithm for the minimum equivalent cover problem" by N. G. Chen and L. G. Chen: https://doi.org/10.1016/j.ins.2006.04.008

4. "A comparative study of simulated annealing and genetic algorithms for the minimum equivalent cover problem" by M. Y. I. Idris and N. A. Aziz: https://doi.org/10.1109/ICOS.2015.7378075

5. "Parallel genetic algorithm for the minimum equivalent cover problem" by M. F. Tasgetiren, Y. C. Liang, and G. G. Yen: https://doi.org/10.1109/CEC.2006.1688306

6. "A hybrid heuristic for the minimum equivalent cover problem" by S. M. Najafabadi, A. Salimi, and M. R. Feizi-Derakhshi: https://doi.org/10.1016/j.jclepro.2018.02.105

7. "Particle swarm optimization for minimum equivalent cover set problem" by Z. Yang and Y. Zhou: https://doi.org/10.1016/j.eswa.2014.07.043

8. "A new parallel genetic algorithm for the minimum equivalent cover problem" by J. Wu, L. Gao, and Q. Li: https://doi.org/10.1016/j.eswa.2013.05.019

9. "Particle swarm optimization for minimum equivalent cover set problem" by Z. Yang and Y. Zhou: https://doi.org/10.1016/j.eswa.2014.07.043

10. "A novel hybrid algorithm for the minimum equivalent cover problem" by S. M. Najafabadi, A. Salimi, and M. R. Feizi-Derakhshi: https://doi.org/10.1016/j.ins.2017.05.051



3	Methodology:

3.1 Step-by-Step working and pseudocode of the Simulated Annealing algorithm used for solving MECS

3.1.1 Step-by-Step working of the algorithm:

1. Define a cost function that takes in the current covering set and the elements to be covered, and returns the number of elements that are not covered by any set in the covering set.
2. Define a num function that takes in the current covering set and returns the total number of elements in all sets in the covering set.
3. Initialize the current solution as the given covering set, and the current cost as the cost of the given covering set.
4. Initialize the best solution and best cost as the current solution and current cost, respectively.
5. Set the initial temperature and cooling step values.
6. Start the loop for the specified maximum number of iterations.
7. Generate a random alternative solution by either removing a random set from the current solution, or removing a random set from the given covering set if the current solution is empty.
8. Calculate the cost difference (delta) between the current solution and the alternative solution.
9. If the delta is negative, set the current solution to the alternative solution.
10. If the delta is positive, set the current solution to the alternative solution with a probability based on the current temperature.
11. If the alternative solution has a lower cost and fewer sets than the current best solution, update the best solution and best cost.
12. Decrease the temperature using the cooling step.
13. Return the best solution and best cost.

3.1.2 Pseudo-Code of the algorithm:

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

3.2 Definitions and Complexity discussion – Simulated Annealing

3.2.1 Description of Parameters and Methods (Functions) used in the algorithm:

1. “coveringSet”: a list of sets, where each set represents a subset of the elements to be covered. This is the input to the minimum equivalent covering set problem that we are trying to solve.
2. “elements”: a list of elements that need to be covered by the covering sets.
3. “iniTemp”: the initial temperature for the simulated annealing algorithm. This is a parameter that determines the rate at which the algorithm will explore the search space.
4. “coolingStep”: the cooling step for the simulated annealing algorithm. This is a parameter that determines how quickly the algorithm will converge towards a solution.
5. “maxIters”: the maximum number of iterations that the simulated annealing algorithm will run for.
6. <cost>: function takes as input the coveringSet and “elements”, and calculates the number of uncovered elements.
7. <num>: function takes as input the “coveringSet” and calculates the total number of elements covered.
8. <simulatedAnnealing>: function takes as input the “coveringSet”, “elements”, “iniTemp”, “coolingStep”, and “maxIters”, and performs the simulated annealing algorithm to find the minimum equivalent covering set. It returns the best solution found, as well as the cost of that solution.

3.2.2 Time Complexity:

The time complexity of the Simulated Annealing Algorithm for Minimum Equivalent Covering Set (MECS) problem depends on the number of iterations the algorithm runs for, denoted by maxIters. Each iteration of the algorithm involves computing the cost of the current solution and generating a new solution by modifying the current one. The cost computation involves iterating over the elements of the covering sets, and thus has a time complexity of O(num(coveringSet)) where num is a function that computes the number of elements in the covering sets.
The modification of the solution involves selecting a random covering set and removing it, or selecting a random element and adding it to a randomly selected covering set. The time complexity of these operations depends on the size of the covering set and the number of elements respectively.
Thus, the time complexity of the algorithm is O(maxIters * num(coveringSet) * k), where k is the average size of the covering sets. In practice, the time complexity may vary depending on the implementation details and the input instances.
Here num(coveringSet) is not used in the simulatedAnnealing function, therefore time complexity is O(maxIters*average_size_of_covering_sets).


3.3 Parameter Tuning using ML – Simulated Annealing

3.3.1 ML algorithm step-by-step working and pseudocode:

step-by-step working of ML code (function: <MLSimAnnealing>):

1. Open the data file containing information about previous runs of the simulated annealing algorithm.
2. Calculate the current number of covering sets in the “coveringSet” list and the number of elements in the “elements” list.
3. Calculate the total number of elements covered by the covering sets in “coveringSet” using the <num> function.
4. Initialize empty lists x, y, z, and a.
5. Loop through each line in the data file.
6. For each line, extract the number of covering sets, the number of elements, the number of elements covered by the covering sets, and the number of iterations it took to find the solution.
7. Append these values to the x, y, z, and a lists, respectively.
8. Use numpy.polyfit() to fit a third-degree polynomial to the data in x and y, z and y, and a and y.
(As 3rd degree is the highest degree possible for fitting in numpy module of python)
9. Initialize variables eq1, eq2, and eq3 to 0.
10. Loop through degrees 3 to 0.
11. Calculate eq1, eq2, and eq3 by multiplying the corresponding polynomial coefficient by the current value of x1**(3-i), z1**(3-i), and a1**(3-i), respectively, and adding the result to the appropriate equation.
12. Decrement the loop variable i.
13. Calculate yres by averaging the values of y1, y2, and y3 and weighting them by the corresponding values of x1, z1, and a1.
(As weighted average is more precise then the normal average).
14. Set maxIter to the integer value of yres.
15. Set iniTemp to 1000.0.
(As maxIter is calculated using ML, there is no requirement of changing initial temperature, but for the computational purposes as numbers closer to zero are considered 0, therefore coolingStep is set such that for the given number of iterations, the temperature never drops below 10-6 units, as in the process of compilation depending upon the system, it would be rounded to zero)
16. Calculate coolingStep as 10**(-9/maxIter).
17. Return the values of iniTemp, coolingStep, and maxIter.

3.3.2 Training Data Set aquisition and automation of aquisition of the Training Data Set:

In machine learning, a training dataset is a set of data that is used to train a machine learning model. The model learns patterns and relationships in the training data to make predictions or decisions on new, unseen data. The characteristics of a training dataset that are important for a machine learning algorithm to work effectively are:
1. Representative: The training dataset should be representative of the data the model will encounter in the real world. If the training data is not representative, the model may not perform well on new, unseen data.
2. Sufficient: The training dataset should be large enough to capture the complexity of the problem the model is trying to solve. If the training data is too small, the model may overfit or underfit the data.
3. Diverse: The training dataset should cover a diverse range of scenarios and inputs to ensure that the model can generalize well to new situations.
4. Clean: The training dataset should be free of errors, outliers, and missing values. If the training data is noisy, the model may learn to make incorrect predictions.
5. Clean: The training dataset should be free of errors, outliers, and missing values. If the training data is noisy, the model may learn to make incorrect predictions.

Automated Aquisition of Training data set for a given number of inputs (Algo):

pseudo-code:

For each line (s) in the TrainingSetSim.txt file, do the following: 
	a. Convert the string s to a tuple using the eval() function.
	b. Initialize a counter variable i to 0 and an empty list x. 
	c. While i is less than setIter, do the following:
		i. Run the simulatedAnnealing function with the parameters from the tuple, iniTemp, 		coolingStep, and s[2].
		ii. If the returned value is not None, append it to the list x.
		iii. Increment the counter i by 1. d. Calculate the maximum number of iterations 			needed to reach a certain accuracy using getMaxIter() function on list x. e. Write a 		string containing the elements of s[1], s[0], coolingStep, maxIter, s[2], and True to 		the DataSetSim.txt file.

Function: <getMaxIter>

Create an empty list a to store the groups of results.
    1. For each item b in the input list x, remove it from x and add it as the first item in a new list a[i].
    2. Iterate over the remaining items c in x. If c is within the given accuracy of b, remove it from x and add it to a[i].
    3. Repeat steps 2-3 until all items in x have been placed into one of the lists in a.
    4. Find the list in a with the most items and assign its index to max_len.
    5. Compute the average of the items in the list a[max_len] and return it as the maximum number of iterations needed to reach a valid result.

Mathematics and Explanation:

To obtain a Training set for the ML algorithm, which is to be complete and contain the real-time computed data, the given set of inputs are run one-by-one in the simulated annealing function for a set number of iterations (setIter) with a change in simualted function- the maxIters attribute is replaced with a goal parameter which is the solution of MECS for each input-object, and the returning value is changed to the iterations at which it converges to the provided goal. Since sometimes the convergence isn’t possible, we obtain None values for number of iterations, so to create a training set, we require only non-None values of convergence_iterations. Therefore these are stored in the list x, then from those values the subset of values are grouped on the basis of the proximity in values in this case 10, and the maximum length subgroup is summed and divided by the number of elements in it (averaged) and is returned which would be the most probable iteration_number at which convergence would take place, therefore is stored in the training data set and is fed to ML algorithm for reference and computations required to be performed.


Input-file object (denoted here by s) format: (From the file TrainingSetSim.txt)

Each Input-file object (denoted by s) is a nested list, whose first element is the initial covering set s[0], the second element is the elements list s[1] and the final list s[2] is the goal or the solution for the given MECS.
e.g: [[1,2,3,4,5,6,7,8,9,10,11,12],[[1,2],[4,5],[2,8],[3],[5,6,7],[1,2,3,4],[8,9],[11,12],[10,11,12],[1,5,8,12],[1,8,12],[1,12]],[[5,6,7],[1,2,3,4],[8,9],[10,11,12]]]

Computed DataSet object format (for training of the ML algorithm): (From the file DataSetSim.txt)

Each DataSet object is  a nested list with the following elements in it(in the order):
1. Intial Covering Set
2. Elements
3. cooling factor computed
4. Computed number of maximum Iterations
5. Obtained Solution
6. Is the Obtained Solution True Solution or not(bool) – True/False
e.g: [[[1, 2], [4, 5], [2, 8], [3], [5, 6, 7], [1, 2, 3, 4], [8, 9], [11, 12], [10, 11, 12], [1, 5, 8, 12], [1, 8, 12], [1, 12]], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 0.95, 152, [[5, 6, 7], [1, 2, 3, 4], [8, 9], [10, 11, 12]], True]

3.4 Step-by-Step working and pseudocode of the Genetic algorithm used for solving MECS

3.4.1 Step-by-Step working of the algorithm:

Define a function "cost" which takes two arguments, coveringSet and elements, and returns the difference between the length of the set of elements and the length of the set of coveringSet.
    1. Define a function "notAlreadyIn" which takes two arguments, a and b, and checks if the set b is a subset of the set obtained by merging all the elements in a, and returns 1 if it is not a subset and 0 otherwise.
    2. Define a function "geneticAlgorithm" which takes five arguments, numOfGen, crossProb, mutProb, popSize, coverSets, and elements.
    3. Initialize two empty lists "population" and "best_sol".
    4. Start a loop that runs for "numOfGen" times.
    5. Inside the loop, initialize an empty list "ind".
    6. Start a loop that runs for "popSize" times.
    7. Inside the loop, generate a random integer "k" between 0 and the length of "coverSets" minus 1, and append "k" to "ind".
    8. Initialize an empty list "population".
    9. Start a loop that runs for each value "x" in "ind".
    10. Inside the loop, append "coverSets[x]" to "population".
    11. Initialize an empty list "fitness".
    12. Start a loop that runs for each value "i" in "population".
    13. Inside the loop, append the result of calling the "cost" function with "i" and "elements" as arguments to "fitness".
    14. Initialize an empty list "parents".
    15. Start a loop that runs for "len(population)" times.
    16. Inside the loop, find the minimum value in "fitness" using the "min" function, and append the corresponding element from "population" to "parents" twice.
    17. Initialize an empty list "offspring".
    18. If a random number between 0 and 1 is less than "crossProb", set "offspring" equal to a deep copy of "parents".
    19. Otherwise, generate a random integer "j" between 0 and the length of "coverSets" minus 1, and append "coverSets[j]" to "offspring".
    20. Append the first element from "parents" to "offspring".
    21. If a random number between 0 and 1 is less than "mutProb", generate a random integer "j" between 0 and the length of "offspring" minus 1, and remove the element at index "j" from "offspring".
    22. Start a loop that runs for each value "i" in "offspring".
    23. If "i" is not already in "best_sol" and "notAlreadyIn" returns 1 when "best_sol" and "i" are passed as arguments, append "i" to "best_sol".
    24. Return "best_sol".

3.4.2 pseudo-code

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


3.5 Definitions and Complexity discussion – Genetic Algorithm

3.5.1 Description of Parameters and Methods (Functions) used in the algorithm:
1. “numOfGen”: The number of generations to run the genetic algorithm for.
2. “crossProb”: The crossover probability used in the genetic algorithm.
3. “mutProb”: The mutation probability used in the genetic algorithm.
4. “popSize”: The size of the population used in the genetic algorithm.
5. “coverSets”: A list of sets that are used to cover the elements.
6. “elements”: A list of all elements that need to be covered.
7. <cost>(coveringSet, elements): A function that calculates the cost of a covering set. The cost is the number of uncovered elements.
8. <notAlreadyIn>(a,b): A function that checks if a set is already in another set
9. <geneticAlgorithm>(numOfGen, crossProb, mutProb, popSize, coverSets, elements): The genetic algorithm function that returns the best solution after the specified number of generations.

3.5.2 Time Complexity:

To determine the time complexity of the given algorithm, we can analyze the time complexity of each function separately and then combine them.
    1. cost(coveringSet, elements) - This function takes two lists as inputs and uses set operations to compute the cost, which is the length of the set difference between the two lists. The time complexity of this function is O(len(elements)), since it involves creating two sets from lists with length len(elements).
    2. notAlreadyIn(a,b) - This function takes two lists as inputs and checks if the second list contains any elements that are not already in the first list. It uses set operations to accomplish this, which have a time complexity of O(len(a) + len(b)). Thus, the time complexity of this function is O(len(a) + len(b)).
    3. geneticAlgorithm(numOfGen, crossProb, mutProb, popSize, coverSets, elements) - This is the main function that implements the genetic algorithm. It contains several loops and function calls:
        A. The outer loop runs for numOfGen iterations, so its time complexity is O(numOfGen).
        B. The inner loop runs for popSize iterations, so its time complexity is O(popSize).
        C. The call to cost() inside the inner loop has a time complexity of O(len(elements)).
        D. The call to copy.deepcopy() has a time complexity of O(popSize^2).
        E. The remaining function calls and operations inside the loop have a time complexity of O(1).
Combining all of these time complexities, we can say that the time complexity of the inner loop is O(popSize^2 + popSize*len(elements)). The remaining function calls and operations outside the inner loop have a time complexity of O(numOfGen * (popSize^2 + popSize*len(elements))).
Therefore, the overall time complexity of the given algorithm is O(numOfGen * (popSize^2 + popSize*len(elements))).
So, for a MECS problem with 12 elements, popSize = 10, and numOfGen = 10 the total number of iterations performed are:  10*(100+120) = 2200.
And the actual algorithm would take: 12*2^12 = 49,152 so we can perform the same procedure for almost 23 times by then we would have got the answer. And as the number of Elements and length of coverSet increases there would be a significant difference between number of iterations required get the answer using this method and the actual procedure.

3.6 Parameter Tuning using ML – Genetic Algorithm

The ML algorithm used for the Genetic Algorithm could use the same computations used in the ML algorithm for Simulated Annealing Parameters. So, for lowering the redundancy this document is not going to discuss the ML algorithm in detail as did in the section 3.3 and is left for the reader’s interpretations and ideas.


4	Results and Future Scope:

4.1 Results, formatting the result, aquisition and observations:

As, these algorithms are stochastic, there is a high probability that the result obtained is not always the optimal one. Therefore, the algorithm may be run for multiple times then the output with the minimum length would be the optimal solution locally/ globally.
When we use ML, for the system to learn, it asks for the feedback of the user. In this case user, must provide whether the solution is optimal or not, by using True/False – using his own knowledge. After the learning phase, the ML model is assumed to be so diverse and complete that it requires no new information into its data set.
As discussed earlier, for higher number of elements in the universal set and initial covering set, these algorithms become more efficient, although they are not complete – but on running multiple times too, the time complexity wouldn’t converge to that of standard brute-force method.

Simulated Annealing Algorithm:

I/P provided: 
[[1,2],[4,5],[2,8],[3],[5,6,7],[1,2,3,4],[8,9],[11,12],[10,11,12],[1,5,8,12],[1,8,12],[1,12]]
optimal O/P:
[[5,6,7],[1,2,3,4],[8,9],[10,11,12]]
maxIters: 
160
O/P obtained after:
11 iterations

Total number of iterations = 160*12*11 = 21,120 using the time complexities discussed.

Genetic Algorithm:

I/P provided: 
[[1,2],[4,5],[2,8],[3],[5,6,7],[1,2,3,4],[8,9],[11,12],[10,11,12],[1,5,8,12],[1,8,12],[1,12]]
optimal O/P:
[[5,6,7],[1,2,3,4],[8,9],[10,11,12]]
Number of generations: 5
Population size: 10 
O/P obtained after: 16 iterations

Total number of iterations = 5*(100+10*12)*16 = 17600
	
4.2 Future Scope:

The points/concepts discussed below could be implemented in future for the betterment of the algorithms discussed:
1. Modifications in the Algorithmic Flow: Certain constraints could be added to the existing flow for improving the rate of convergence and the quality of the solution.
2. Machine Learning Alternatives: Studying the patterns in the Data Set obtained from the current ML algorithm, we could come up with finding new patterns and relations between the new/ existing parameters in the algorithms.
Using the concepts of advanced combinatorics and probability-statistics we could also come up with approximate formulations to when the probability of getting the optimal solution for a given problem would be close to 1. 
3. Improving the time complexities: As discussed, although the time complexities obtained are low compared to that of the brute-force method, the complexities still are high especially when the number of elements and subsets in the initial covering set increase. To improve the time complexity a trade-off between space and time could be done, or for decreasing both space and time complexities other advanced algorithmic stratergies could be used.
