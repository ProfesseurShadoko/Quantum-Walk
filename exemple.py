from CTQW import tWalk2D

walk = tWalk2D(7)
walk.particle = walk.fock(3,3)

for state in walk.positions(): #state = (0,0), then (0,1), then ...
    x,y=state
    for neighbour in walk.neighbours(x,y): #neighbourg of (0,0) are (0,1) and (1,0)
        n_x,n_y=neighbour
        walk[x,y,n_x,n_y] = 1

print(repr(walk))

walk.set_animation_param(500,1000)
walk.run()
