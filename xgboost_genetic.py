#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mohit jain
"""

'''
We will use genetic algorithum to optimize hyperparameters for XGboost. 
'''

# Importing the libraries
import numpy as np
import pandas as pd
import geneticXGboost #genetic algorithum module
import xgboost as xgb


np.random.seed(723)
'''
The dataset is from https://archive.ics.uci.edu/ml/machine-learning-databases/musk/
It contains a set of 102 molecules, out of which 39 are identified by humans as 
having odor that can be used in perfumery and 69 not having the desired odor.
The dataset contains 6,590 low-energy conformations of these molecules, contianing 166 features.
'''

# Importing the dataset
dataset = pd.read_csv('clean2.data', header=None)

X = dataset.iloc[:, 2:168].values #discard first two coloums as these are molecule's name and conformation's name

y = dataset.iloc[:, 168].values #extrtact last coloum as class (1 => desired odor, 0 => undesired odor)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 97)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#XGboost Classifier

#model xgboost
#use xgboost API now
xgDMatrix = xgb.DMatrix(X_train, y_train) #create Dmatrix
xgbDMatrixTest = xgb.DMatrix(X_test, y_test)


'''
Let's find optimized parameters using genetic algorithms
'''

numberOfParents = 8 #number of parents to start
numberOfParentsMating = 4 #number of parents that will mate
numberOfParameters = 7 #number of parameters that will be optimized
numberOfGenerations = 4 #number of genration that will be created

#define the population size

populationSize = (numberOfParents, numberOfParameters)

#initialize the population with randomly generated parameters
population = geneticXGboost.initilialize_poplulation(numberOfParents)

#define an array to store the fitness  hitory
fitnessHistory = np.empty([numberOfGenerations+1, numberOfParents])

#define an array to store the value of each parameter for each parent and generation
populationHistory = np.empty([(numberOfGenerations+1)*numberOfParents, numberOfParameters])

#insert the value of initial parameters to history
populationHistory[0:numberOfParents, :] = population

for generation in range(numberOfGenerations):
    print("This is number %s generation" % (generation))
    
    #train the dataset and obtain fitness
    fitnessValue = geneticXGboost.train_population(population=population, dMatrixTrain=xgDMatrix, dMatrixtest=xgbDMatrixTest, y_test=y_test)
    fitnessHistory[generation, :] = fitnessValue
    
    #best score in the current iteration
    print('Best F1 score in the this iteration = {}'.format(np.max(fitnessHistory[generation, :])))

    #survival of the fittest - take the top parents, based on the fitness value and number of parents needed to be selected
    parents = geneticXGboost.new_parents_selection(population=population, fitness=fitnessValue, numParents=numberOfParentsMating)
    
    #mate these parents to create children having parameters from these parents (we are using uniform crossover)
    children = geneticXGboost.crossover_uniform(parents=parents, childrenSize=(populationSize[0] - parents.shape[0], numberOfParameters))
    
    #add mutation to create genetic diversity
    children_mutated = geneticXGboost.mutation(children, numberOfParameters)
    
    '''
    We will create new population, which will contain parents that where selected previously based on the
    fitness score and rest of them  will be children
    '''
    population[0:parents.shape[0], :] = parents #fittest parents
    population[parents.shape[0]:, :] = children_mutated #children
    
    populationHistory[(generation+1)*numberOfParents : (generation+1)*numberOfParents+ numberOfParents , :] = population #srore parent information
    

#Best solution from the final iteration

fitness = geneticXGboost.train_population(population=population, dMatrixTrain=xgDMatrix, dMatrixtest=xgbDMatrixTest, y_test=y_test)
fitnessHistory[generation+1, :] = fitness

#index of the best solution
bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]

#Best fitness
print("Best fitness is =", fitness[bestFitnessIndex])

#Best parameters
print("Best parameters are:")
print('learning_rate', population[bestFitnessIndex][0])
print('n_estimators', population[bestFitnessIndex][1])
print('max_depth', int(population[bestFitnessIndex][2])) 
print('min_child_weight', population[bestFitnessIndex][3])
print('gamma', population[bestFitnessIndex][4])
print('subsample', population[bestFitnessIndex][5])
print('colsample_bytree', population[bestFitnessIndex][6])


#visualize the change in fitness of the various generations and parents


geneticXGboost.plot_parameters(numberOfGenerations, numberOfParents, fitnessHistory, "fitness (F1-score)")

#Look at individual parameters change with generation
#Create array for each parameter history (Genration x Parents)

learnigRateHistory = populationHistory[:, 0].reshape([numberOfGenerations+1, numberOfParents])
nEstimatorHistory = populationHistory[:, 1].reshape([numberOfGenerations+1, numberOfParents])
maxdepthHistory = populationHistory[:, 2].reshape([numberOfGenerations+1, numberOfParents])
minChildWeightHistory = populationHistory[:, 3].reshape([numberOfGenerations+1, numberOfParents])
gammaHistory = populationHistory[:, 4].reshape([numberOfGenerations+1, numberOfParents])
subsampleHistory = populationHistory[:, 5].reshape([numberOfGenerations+1, numberOfParents])
colsampleByTreeHistory = populationHistory[:, 6].reshape([numberOfGenerations+1, numberOfParents])

#generate heatmap for each parameter

geneticXGboost.plot_parameters(numberOfGenerations, numberOfParents, learnigRateHistory, "learning rate")
geneticXGboost.plot_parameters(numberOfGenerations, numberOfParents, nEstimatorHistory, "n_estimator")
geneticXGboost.plot_parameters(numberOfGenerations, numberOfParents, maxdepthHistory, "maximum depth")
geneticXGboost.plot_parameters(numberOfGenerations, numberOfParents, minChildWeightHistory, "minimum child weight")
geneticXGboost.plot_parameters(numberOfGenerations, numberOfParents, gammaHistory, "gamma")
geneticXGboost.plot_parameters(numberOfGenerations, numberOfParents, subsampleHistory, "subsample")
geneticXGboost.plot_parameters(numberOfGenerations, numberOfParents, colsampleByTreeHistory, "col sample by history")

