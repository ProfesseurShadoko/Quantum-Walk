
# Quantum Walks

Using the QuTiP library in Python, I designed a bunch of simulations for the study of Quantum Random Walks. It is still a work in progress. Here's what it looks like !

## DTQW

DTQW are Discrete-Time Quantum Walks, for an n-step walk, an operator is applied n times to a particle, and then the position of the particle is mesured.
To garantee the *randomness* of the walk, usually a *coin* operator is applied on each iteration. You can find out what it looks like in the folder **coin_walks**, but it is not what this project is focused on.
You can also find an implementation of the Grover Algorithm, which can be seen as a Quantum Walk.

## CTQW

CTQW are Continous-Time Quantum Walks. The idea here is to let a particle evolve on a lattice, with tunneling probabilities between the different sites of the lattice. This is what this project is focused on, and this is what it looks like !

### An exemple

Let's say you want to perform a 2D-CTQW on a lattice of size 7x7. First we want to initialize the Walk and put the particle in the middle of the Lattice, on position (3,3).

```python
from CTQW import tWalk2D

walk = tWalk2D(7)
walk.particle = walk.fock(3,3)
```

Now, we need to define the Hamiltonian that we want to apply on the particle. Let's keep it simple, let's say we want, for each site of the lattice, the same coupling with its neighbours.

```python

for state in walk.positions(): #state = (0,0), then (0,1), then ...
    x,y=state
    for neighbour in walk.neighbours(x,y): #neighbourg of (0,0) are (0,1) and (1,0)
        n_x,n_y=neighbour
        walk[x,y,n_x,n_y] = 1
```

Of course, the hamiltonian isn't necessary hermitian or unitary, but this will be corrected when the simulation will start. Let's see where we are right now !

```python
print(repr(walk))
```
```
Walk of type tWalk2D :
                 - Dimension of the walk : 2D
                 - Size of the grid : 7x7
                 - Space size : 49
                 - Hamiltonian size : 2401
                 - Walk solved : False
                 - Parameters initialized : False
```

As we can see, we have to initialize some parameters, the duration of the simulation and the number of frames. When the walk will be resolved, the simulation will run based on the following discretization of time :
```python
time = np.linspace(0,duration,frame_nbr)
```
Once we have defined those value, we can launch the simulation !
```python
walk.set_animation_param(duration=500,frame_nbr=1000)
walk.run()
```

And that's it ! You can run the script exemple.py to see what it looks like !

***

But now, let's say you want the same in 3D, on a 15x15x15 grid.

```python
from CTQW import tWalk3D

walk = tWalk3D(15)
walk.particle = walk.fock(7,7,7)
walk.set_animation_param(60_000,1000)

print(repr(walk))
```
```
Walk of type tWalk3D :
                 - Dimension of the walk : 3D
                 - Size of the grid : 15x15x15
                 - Space size : 3375
                 - Hamiltonian size : 11390625
                 - Walk solved : False
                 - Parameters initialized : True
```

As you can see, the hamiltonian is hudge, and saddly the __setitem__ method of the walk has a very bad complexity, because it copies the hamiltonian, wich means we have to avoid using walk[i,j]=...
The right way is to define the hamiltonian first as a np.ndarray, and then transform it into a Qobj. But how can we see wich coefficient correspond to wich position ? Indeed, we are working here on tensor states |x,y,z>. Luckily, there is a function for that! And tis is how it can be used:

```python
hamiltonian = np.zeros((walk.part_dim,walk.part_dim))  
for state in walk.positions():
    x,y,z=state
    for neighbour in walk.neighbours(x,y,z):
        n_x,n_y,n_z=neighbour
        state_index = walk.to_index(x,y,z)
        neighbour_index = walk.to_index(n_x,n_y,n_z)
        hamiltonian[state_index][neighbour_index]=1

walk.set_hamiltonian(hamiltonian)
```

Finally, the simulation can be launched. But because of the amount of data that we want to represent, the visualization experiences a drop in fps. But there are other ways to get information about the walk !

```python

#first we run the simulation without showing it
walk.solve()

#then we can get information about the distribution of the particle :
walk.part_coeff(x,y,z,t) #returns <x,y,z|particle> at the time t (an integer!)

distribution = walk.get_distribution(t)
distribution[x,y,z] #returns the probability for the position (x,y,z)

walk.mean(t) #average position
walk.variance(t)
walk.std(t) #standard deviation

#show particle at time t
walk.show(time)
```

And that's it ! For now...