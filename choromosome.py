import random

class Chromosome :
    def __init__(self , lenght , function):
        self.genes = ""
        self.function = function
        for i in range(lenght):
            self.genes += str(random.randint(0, 1)) 
        self.calculateTheFitness()    
    
    def calculateTheFitness(self):
        decimalValueOfGenes = self.convertToDecimal()
        fitnessValue = self.function(decimalValueOfGenes)
        self.fitness = fitnessValue
        
    def convertToDecimal(self):
        decimal = sum([pow(2 , i) * int(x) for x,i in zip(self.genes , reversed(range(len(self.genes))))])
        return decimal