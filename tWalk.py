from twalk_base import tWalk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np


class tWalk1D(tWalk):
    
    DIM=1
    x_label="x"
    y_label="probability"
    
    def plot_ax(self, ax: Axes3D, time: int):
        X=[]
        Y=[]
        dist = self.get_distribution(time)
        for pos in self.positions():
            pos=pos[0]
            X.append(pos)
            Y.append(dist[pos])
        ax.plot(X,Y)

class tWalk2D(tWalk):
    
    DIM=2
    x_label="x"
    y_label="y"
    z_label="probability"
    
    def plot_ax(self,ax:Axes3D, time:int):
        X,Y=np.arange(self.size),np.arange(self.size)
        X,Y = np.meshgrid(X,Y)
        Z=self.get_distribution(time)
            
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap="plasma"
        )

class tWalk3D(tWalk):
    
    DIM=3
    x_label="x"
    y_label="y"
    z_label="z"
    cmap="GnBu"
    show_cbar=True
    
    
    def plot_ax(self,ax:Axes3D, time:int):
        X=[]
        Y=[]
        Z=[]
        VAL=[]
        dist=self.get_distribution(time)
        for position in self.positions():
            X.append(position[0])
            Y.append(position[1])
            Z.append(position[2])
            VAL.append(dist[position])
        
        p1=ax.scatter(X,Y,Z,c=VAL,cmap=self.cmap)
        
        if self.show_cbar:
            if hasattr(self,"_cbar"):
                self._cbar.remove()
            self._cbar=plt.colorbar(p1,label="probability")
            
    

if __name__=="__main__":
    WALK=10
    
    def test1D():
        walk = tWalk1D.uniform_tunneling(2*WALK+1)
        walk.set_animation_param(duration=100*WALK,frame_nbr=1000)
        walk.run()
    
    def test2D():
        walk = tWalk2D.uniform_tunneling(2*WALK+1)
        walk.set_animation_param(duration=100*WALK**2,frame_nbr=1000)
        walk.run()
     
    def test3D(): 
        WALK=3
        print("Starting...")
        #walk = tWalk3D.uniform_tunneling(2*WALK+1)
        walk = tWalk3D.random_tunneling(2*WALK+1)
        #walk.particle=walk.fock(1,1,1)
        
        walk.set_animation_param(duration=200*WALK**3,frame_nbr=1000)
        print(walk.__repr__())
        
        print("Solving...")
        walk.run()
    
    test3D()
    
    
    
    
    
    
    