import numpy as np
import matplotlib.pyplot as plt 
import math
import random
from numba import jit
import numba as nb
from itertools import product
import os
import json
import random


#set directory
file_path = ""
os.chdir(file_path)

events0=[(0,0,0,0,0,0,0)]
#Events in the sim
events=[
 #B#R#C#N#P#S#A   
(+1,0,0,0,0,0,0),
(+1,0,0,0,0,0,0),
(-1,0,0,0,0,0,0),

(0,+1,0,0,0,0,0),
(0,-1,1,0,0,0,0),
(0,-1,0,0,0,0,0),

(0,0,-1,0,0,0,0),

(0,0,0,+1,0,0,0),
(0,0,0,-1,0,0,0),
    
(0,0,0,0,+1,0,0),  
(0,0,0,0,-1,0,0),
    
(0,0,0,0,0,+1,0), 
(0,0,0,0,0,-1,0),
    
(0,0,0,0,0,0,+1),
(0,0,0,0,0,0,-1)
]

events = np.asarray(events)
events0 = np.asarray(events0)
#Gillespie algorithm
@jit(nopython=True)
def Gillespie(size, f_t , k0 , b1 , b2, b3, b4, b5, q1, Zn, Zs, 
B1,
R1,
C1,
N1,
P1,
S1,
A1):
    
    #Intializing time steps
    ns   = 0
    time = 0
    
    #Number of events
    no_events = np.arange(0,15,1)
    
    #Vector of t and proteins
    t = np.zeros(size)
    v = np.zeros((size, 7))
    
    #Initial conditions
    v[(0, 0)] = B1
    v[(0, 1)] = R1
    v[(0, 2)] = C1
    v[(0, 3)] = N1
    v[(0, 4)] = P1
    v[(0, 5)] = S1
    v[(0, 6)] = A1
    
    #Run the sim until maxtime
    for ns in range(size-1):
       
        #Rates of reactions
        rates = np.array([f_t, #f(t)
                          k0*v[(ns, 0)], #k0*B
                          v[(ns, 6)]*v[(ns, 0)], #A*B
                          
                          b1* ( v[(ns, 3)] / ( v[(ns, 3)] + Zn) ), #b1 (Nt/ (Nt + Zn))
                          v[(ns, 0)]*v[(ns, 1)], #B*R
                          v[(ns, 4)]*v[(ns, 1)], #P*R
                          
                          v[(ns, 4)]*v[(ns, 2)],#P*C
                          
                          b2*v[(ns, 2)],#b2*C
                          q1*v[(ns, 3)],#q1*N
                          
                          b3 * ( v[(ns, 3)] / ( v[(ns, 3)] + Zn) ),#b3 (Nt/ (Nt + Zn))
                          q1*v[(ns, 4)],#q1*P
                          
                          b4 * ( v[(ns, 3)] / ( v[(ns, 3)] + Zn) ),#b4 (Nt/ (Nt + Zn))
                          q1*v[(ns, 5)],#q1*S
                          
                          b5 * ( v[(ns, 3)] / ( v[(ns, 3)] + Zn + (v[(ns, 5)] * Zn / Zs)  ) ),#b5 (Nt/ (Nt + Zn + (S*Zn /Zs) ))
                          q1*v[(ns, 6)] #q1*A
                          
                         ])
                        
        #Total rate
        total_rate  =  np.sum(rates)
        
        if total_rate == 0:
            v[ns+1] = v[ns]+events0[0]
            ns = ns + 1
        else:

            choose = np.searchsorted(np.cumsum(rates/total_rate), np.random.rand(1))[0]
            
            v[ns+1] = v[ns]+events[choose]
            ns = ns + 1
         
    return v[0:size]

  


#Paramters
size = 1000
k0 = 0.1

#Starting values for 10,000 individuals
b1_vector = [random.uniform(0, 10) for _ in range(10000)]
b2_vector = [random.uniform(0, 10) for _ in range(10000)]
b5_vector = [random.uniform(0, 10) for _ in range(10000)]
q1_vector =  [random.uniform(0, 1) for _ in range(10000)]
b3_vector = [random.uniform(0, 10) for _ in range(10000)]
b4_vector = [random.uniform(0, 10) for _ in range(10000)]
Zn_vector = [random.uniform(0, 10) for _ in range(10000)]
Zs_vector = [random.uniform(0, 10) for _ in range(10000)]

b1_init = b1_vector 
b2_init = b2_vector 
b5_init = b5_vector 
q1_init = q1_vector 
b3_init = b3_vector 
b4_init = b4_vector 
Zn_init = Zn_vector 
Zs_init = Zs_vector 

#Intitial condition
B1 = 1
R1 = 10
C1 = 0
N1 = 0
P1 = 0
S1 = 0
A1 = 0
f_t = 0

#Population size
nPopulation = 10000

nGeneration = 10000

#probability of the mutation
Pmutate     = 0.001

#Strength of the mutation
max_mutate = 1

#counter
x = 0

for _ in range(nGeneration):
    
    #Vector of fitness values for individuals
    fitness_vec = []
    
    #Cacluate fitnes for each individual 
    for i in range(nPopulation):

        b1 = b1_vector[i]
        b2 = b2_vector[i]
        b5 = b5_vector[i]
        q1 = q1_vector[i]
        b3 = b3_vector[i]
        b4 = b4_vector[i]
        Zn = Zn_vector[i]
        Zs = Zs_vector[i]
        
        #Mutations
        
        if random.random() <= Pmutate:
            if random.random() < 0.5:
                b1 += random.uniform(0, max_mutate) 
            else:
                b1 -= random.uniform(0, max_mutate)
                
        if random.random() <= Pmutate:
            if random.random() < 0.5:
                b2 += random.uniform(0, max_mutate)
            else:
                b2 -= random.uniform(0, max_mutate) 
                    
        if random.random() <= Pmutate:
            if random.random() < 0.5:
                b5 += random.uniform(0, max_mutate) 
            else:
                b5 -= random.uniform(0, max_mutate)
                
        if random.random() <= Pmutate:
            if random.random() < 0.5:
                q1 += random.uniform(0, max_mutate) 
            else:
                q1 -= random.uniform(0, max_mutate)  
                
        if random.random() <= Pmutate:
            if random.random() < 0.5:
                b3 += random.uniform(0, max_mutate) 
            else:
                b3 -= random.uniform(0, max_mutate) 
                
        if random.random() <= Pmutate:
            if random.random() < 0.5:
                b4 += random.uniform(0, max_mutate) 
            else:
                b4 -= random.uniform(0, max_mutate)  
                
        if random.random() <= Pmutate:
            if random.random() < 0.5:
                Zn += random.uniform(0, max_mutate)  
            else:
                Zn -= random.uniform(0, max_mutate) 
                
        if random.random() <= Pmutate:
            if random.random() < 0.5:
                Zs += random.uniform(0, max_mutate)  
            else:
                Zs -= random.uniform(0, max_mutate) 
        
        #Loss of function mutations
        if b1 < 0:
            b1 = .000001

        if b2 < 0:
            b2 = .000001
        
        if b5 < 0:
            b5 = .000001
        
        if q1 < 0:
            q1 = .000001
        
        if b3 < 0:
            b3 = .000001
        
        if b4 < 0:
            b4 = .000001
        
        if Zn < 0:
            Zn = .000001
        
        if Zs < 0:
            Zs = .000001
        
        v =  Gillespie(
                  size = size,  
                  f_t = f_t , 
                  k0 = k0, 
                  b1 = b1, 
                  b2 = b2, 
                  b3 = b3, 
                  b4 = b4, 
                  b5 = b5, 
                  q1 = q1, 
                  Zn = Zn, 
                  Zs = Zs,
                  B1 = B1,
                  R1 = R1,
                  C1 = C1,
                  N1 = N1,
                  P1 = P1,
                  S1 = S1,
                  A1 = A1)
        
        #Individual fitness values
        fitness_vec.append( np.exp(-(np.mean(v[:,0]) +np.mean(v[:,1]) +np.mean(v[:,3]) +np.mean(v[:,4]) +np.mean(v[:,5]) +np.mean(v[:,6]) )) )
       
    
    normalized_fitness = [p / sum(fitness_vec) for p in fitness_vec]
    
    b1_vector = np.random.choice(b1_vector, nPopulation, p = normalized_fitness)
    b2_vector = np.random.choice(b2_vector, nPopulation, p = normalized_fitness)
    b5_vector = np.random.choice(b5_vector, nPopulation, p = normalized_fitness)
    q1_vector = np.random.choice(q1_vector, nPopulation, p = normalized_fitness)
    b3_vector = np.random.choice(b3_vector, nPopulation, p = normalized_fitness)
    b4_vector = np.random.choice(b4_vector, nPopulation, p = normalized_fitness)
    Zn_vector = np.random.choice(Zn_vector, nPopulation, p = normalized_fitness)
    Zs_vector = np.random.choice(Zs_vector, nPopulation, p = normalized_fitness)
    
    #print the generation time
    x = x+1
    print(x)
    
data =  {
    
    'b1': b1_vector.tolist(),
    'b2': b2_vector.tolist(),
    'b5': b5_vector.tolist(),
    'q1': q1_vector.tolist(),
    'b3': b3_vector.tolist(),
    'b4': b4_vector.tolist(),
    'Zn': Zn_vector.tolist(),
    'Zs': Zs_vector.tolist(),
    'b1_0': b1_init,
    'b2_0': b2_init,
    'b5_0': b5_init,
    'q1_0': q1_init,
    'b3_0': b3_init,
    'b4_0': b4_init,
    'Zn_0': Zn_init,
    'Zs_0': Zs_init}


with open('mu_001_F1.json', 'w') as json_file:
    json.dump(data, json_file)






