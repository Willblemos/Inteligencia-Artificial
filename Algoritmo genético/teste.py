
# Objetivo: achar um vetor de inteiros (entre i_min e i_max) com i_length posicoes cuja a soma de todos os termos seja o mais proximo possivel de target

#O algoritmo rodara epochs vezes -> numero de populacoes geradas. Sera impresso a media de fitness de cada uma das epochs populacoes

#RODAR COM PYTHON 2!!! (senao colocar () em print e tirar x de xrange
  
from genetic import *
target = 708
i_length = 10
i_min = 0
i_max = 100
p_count = 100
epochs = 120
p = population(p_count, i_length, i_min, i_max)
fitness_history = [media_fitness(p, target),]
for i in range(epochs):
   
    p = evolve(p, target)
    fitness_history.append(media_fitness(p, target))

for datum in fitness_history:
   print (datum)

