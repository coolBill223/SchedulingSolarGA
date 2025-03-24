import random
from population import Population
from choromosome import Chromosome
class GeneticAlgorithm : 
    
    def __init__(self , populationSize , chromosomeSize , tournamentSize , elitismSize , mutationRate , function):
        self.populationSize = populationSize
        self.chromosomeSize = chromosomeSize
        self.tournamentSize = tournamentSize
        self.elitismSize    = elitismSize
        self.mutationRate   = mutationRate
        self.function       = function
    
        
    def reproduction(self , population):
        temp = []
        temp[:self.elitismSize] = population.getNFittestChromosomes(self.elitismSize)
        for i in range(self.elitismSize , self.populationSize):
            parent1 = self.tournamentSelection(population)
            parent2 = self.tournamentSelection(population)
            
            child = self.onePointCrossOver(parent1, parent2)
            
            self.bitFlipMutation(child)
            
            temp.append(child)
            
        newPopulation = Population(self.populationSize, self.chromosomeSize, self.function, False)
        newPopulation.chromosomes = temp
        newPopulation.findTheFittest()
        newPopulation.calculateTheFitnessForAll()
        return newPopulation
        
    
    def bitFlipMutation(self , child):
        if random.random() < self.mutationRate :
            mutationPoint = random.randint(0, len(child.genes) -1)
            geneslist = list(child.genes)
            geneslist[mutationPoint] = "0" if geneslist[mutationPoint] == "1" else "1"
            child.genes = ''.join(geneslist)
            child.calculateTheFitness()
    
    def tournamentSelection(self , population):
        tournamentPool = []
        for i in range(self.tournamentSize):
            index = random.randint(0, len(population.chromosomes) -1)
            tournamentPool.append(population.chromosomes[index])
        tournamentPool.sort(key = lambda x:x.fitness)
        return tournamentPool[0]
    
    def onePointCrossOver(self , parent1 , parent2):
        temp = []
        crossOverPoint = random.randint(0, len(parent1.genes) -1)            
        temp[:crossOverPoint] = parent1.genes[:crossOverPoint]
        temp[crossOverPoint:] = parent2.genes[crossOverPoint:]
        child = Chromosome(7, self.function)
        child.genes = ''.join(temp)
        child.calculateTheFitness()
        return child