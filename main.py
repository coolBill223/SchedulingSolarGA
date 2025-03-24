import matplotlib.pyplot as plt
import numpy as np
from population import Population
from geneticAlgorithm import GeneticAlgorithm

#function to be Optimize        
def f(x):
  return pow(x, 2) + 2 * x - 1
    
    # plot The Function in the interval [-256 , 256] .
x = np.arange(-256,256,1)
plt.plot(x , f(x))
    
    #Generate The   Initial Population
initialPopulation = Population(populationSize = 20, chromosomeSize = 10, function = f, init = True)    
"""
    Create an Instance of Genetic Algorithm with this Parameters : 
        * pop_size = 20 
        * chromosome_Size = 8 (as I mentioned before, we will try to optimize the function f in this range [-256, 256], and 8 bits are enough to encode 256.) 
        * tournament_Pool_size = 4 
        * elitism_size = 4 
        * mutation_rate = 0.1 
        * and Finally the Function to be Optimize
"""
GeneticAlgo = GeneticAlgorithm(populationSize = 20, chromosomeSize = 10, tournamentSize = 4, elitismSize = 5, mutationRate = 0.1, function = f)
    
    
population = initialPopulation
    # repeat The Process 50 times , 50 is the number of generations .
for i in range(50):    
    x_i = population.fittest.convertToDecimal()
    f_x_i = f(x_i)
    plt.scatter(x_i, f_x_i, color='red')
    population = GeneticAlgo.reproduction(population)