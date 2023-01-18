from qutip import Qobj, tensor, fock,qeye
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import pickle

class Grover:
    """see : https://www.youtube.com/watch?v=9uWiExuEndY
    """
    
    #create Hermitian space
    def __init__(self,size:int=10):
        self.size = size
        self.target = 630#np.random.randint(size)
        self.particle = sqrt(1/self.size)*sum([self.fock(i) for i in range(size)])
        self.iteration_number = int(np.pi/4*sqrt(self.size))*2
    
    def __str__(self)->str:
        return f"<Grover instance : size={self.size}, target={self.target}, iterations={self.iteration_number} >"
    
    def __repr__(self)->str:
        return f"Grover_algo of size {self.size}"
    
    def fock(self,n:int):
        return fock(self.size,n)
    
    @property
    def oracle(self)->Qobj: #coin !
        """returns I-2|target><targe| #flips target"""
        if not hasattr(self,"_oracle"):
            self._oracle=qeye(self.size)-2*(self.fock(self.target)*self.fock(self.target).dag())
        return self._oracle
    
    @property
    def mean_inversion(self)->Qobj: #shift !
        if not hasattr(self,"_mean"):
            psi_h:Qobj = sqrt(1/self.size)*sum([self.fock(i) for i in range(self.size)]) #moyenne
            self._mean = 2*psi_h*psi_h.dag()-qeye(self.size) #symetrie par rapport à la moyenne
        return self._mean
    
    #run simulation
    def apply_gate(self):
        if not hasattr(self, "time_evolution"):
            self.time_evolution=[self.particle]
        self.particle = self.mean_inversion*self.oracle*self.particle
        self.time_evolution.append(self.particle)
        
        
    def solve(self)->None:
        assert not hasattr(self,"time_evolution"),"Algorithm has already been used"
        for i in range(self.iteration_number):
            print(f"\rSolving... {i/self.iteration_number:.2%}",end="")
            self.apply_gate()
        print("\rSolving... 100%           ")
    
    #display simulation
    def part_coeff(self,x:int,t:int=None)->complex:
        if t!=None:
            part = self.time_evolution[t]
        else:
            part = self.particle
        return part[x][0]
    
    def get_distribution(self,t:int)->list:
        return [abs(self.part_coeff(x,t))**2 for x in range(self.size)]
    
    def run(self)->None:
        if not hasattr(self,"time_evolution"):
            self.solve()
            
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        def animate(t):
            if t>=len(self.time_evolution):
                return 
            
            fig.suptitle(f"Probabilité de détection après {min(t,len(self.time_evolution)-1)+1} étapes")
            ax.clear()
            ax.set_xlabel("probabilité")
            ax.set_ylabel("position")
            X = [n for n in range(self.size)]
            ax.plot(
                X,
                self.get_distribution(t),
            )
            ax.scatter([self.target],[0],c="red",label="cible")
            ax.set_ybound(-0.2,1.2)
            ax.legend()
        
        ani = anim.FuncAnimation(fig,animate,interval=10)
        plt.show()

def save(grover,):
    with open(f"{grover.__repr__()}.psc","wb") as file: #write bytes
        pickle.dump(grover, file)

def load(filename:str):
    with open(f"{filename}.psc","rb") as file: #write bytes
        return pickle.load(file)


if __name__=="__main__":
    N=1000
    grover = Grover(N)
    grover.run()
    
    