

from random import randint,choice, random 
from operator import add
from functools import reduce

def individual(length, min, max):
    'Create a member of the population.'
    return [ randint(min,max) for x in range(length) ]

def population(count, length, min, max):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [ individual(length, min, max) for x in range(count) ]

def fitness(individual, target):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    target: the target number individuals are aiming for

    O fitness do individuo perfeito sera ZERO, ja que o somatorio dara o target
    reduce: reduz um vetor a um escalar, neste caso usando o operador add
    """
    sum = reduce(add, individual, 0)
    return abs(target-sum)

def media_fitness(pop, target):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x, target) for x in pop))
    return summed / (len(pop) * 1.0)



def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.2):
    'Tabula cada individuo e o seu fitness'
    graded = [ (fitness(x, target), x) for x in pop]
    aux = sorted(graded) #salvando informação
    
    'Ordena pelo fitness os individuos - menor->maior'
    graded = [ x[1] for x in sorted(graded)]
    'calcula qtos serao elite'
    retain_length = int(len(graded)*retain)
    'elites ja viram pais'

    parents = graded[:retain_length]  #elites 20%

    aux = aux[retain_length:] #separa os 80% restantes

    for individual in aux: #torneio
        if random_select > random():
            sorteio = []
            chosen = []
            # print("\n\n *****GERAÇÃO:" + str(aux))
            for x in range(2):
                lutador = choice(aux)
                sorteio.append(lutador)
            # print("\nLUTADORES:" + str(sorteio))
            chosen = [ x[1] for x in sorted(sorteio) ]
            parents.append(chosen[0])
            # print("\nVENCEDOR:" + str(chosen[0]))
            chosen.clear()
            sorteio.clear()
            # print("\n *****PARENTS: *****"+str(parents))
        
    for individual in parents:
        if mutate > random(): #mutate some individuals
            pos_to_mutate = randint(0, len(individual)-1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = randint(
                min(individual), max(individual))
    # crossover parents to create children
    parents_length = len(parents)
    'descobre quantos filhos terao que ser gerados alem da elite e aleatorios'
    desired_length = len(pop) - parents_length
    children = []
    'comeca a gerar filhos que faltam'
    while len(children) < desired_length:
        'escolhe pai e mae no conjunto de pais'
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) // 2
            'gera filho metade de cada'
            child = male[:half] + female[half:]
            'adiciona novo filho a lista de filhos'
            children.append(child)
    'adiciona a lista de pais (elites) os filhos gerados'
    parents.extend(children)
    aux.clear()
    return parents