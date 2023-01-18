from qutip import Qobj,tensor,qeye,fock
import numpy as np
import matplotlib.pyplot as plt
from shadok.progress_bar import ProgressIterator


class Walk:
    steps:int=100

    def __init__(self,start:Qobj, coin:list[list]):
        self.particle = start/start.norm()
        self.start = start/start.norm()

        self.coin = np.array(coin)
        self.distribution=[]
        
    
    def make_step(self):
        self.particle = self.step_operator()*self.coin_operator()*self.particle
        self.particle=self.particle/self.particle.norm()
    
    def run(self,show:bool=True,show_progress=True):
        """
        Runs the walk to the end

        Calls make_step(Walk.steps-self.distance) and shows the result if asked

        Args:
            show : defines if, at the end of the walk, the result shall be shown

        """
        if show_progress:
            for i in ProgressIterator(range(Walk.steps),message="Performing Walk..."):
                self.make_step()
        else:
            for i in range(Walk.steps):
                self.make_step()
        
        if show:
            print(f"Espérance : {self.mean()}\nEcart type : {self.std()}\nCardinal : {int(self.cardinal()*100)}%\nDiversity : {int(self.diversity()*100)}%")
            self.show()
        
    #operators
    def step_operator(self)->Qobj:
        """
        Returns:
            the operator that, applied to the current particle, allow the walk to make one more step
        """
        return tensor(Walk.spin("+")*Walk.spin("+").dag(),self.move_up())-tensor(Walk.spin("-")*Walk.spin("-").dag(),self.move_down())
    
    def coin_operator(self)->Qobj:
        """
        Returns the coin that resets the spin
        """
        return tensor(Qobj(self.coin),qeye(2*Walk.steps+1))

    def move_up(self)->Qobj:
        out = np.zeros((2*Walk.steps+1,2*Walk.steps+1))
        for i in range(2*Walk.steps):
            out[i,i+1]=1.0
        return Qobj(out)

    def move_down(self)->Qobj:
        out = np.zeros((2*Walk.steps+1,2*Walk.steps+1))
        for i in range(1,2*Walk.steps+1):
            out[i,i-1]=1.0
        return Qobj(out)
    
    #mesure
    def get_coeff(self,n:int):
        """
        Args:
            n : position for whom you want to know the coefficient
        
        Returns:
            coefficient of position n (spin(+) + spin(-))
        """
        n=n+Walk.steps
        return (tensor(qeye(2),fock(Walk.steps*2+1,n)*fock(Walk.steps*2+1,n).dag())*self.particle).norm()
    
    def get_distribution(self):
        """
        Creates the distribution of probability for each position

        Returns:
            list of probabilities from -Walk.steps to Walk.steps
        """
        if self.distribution==[] or self.distribution==None:
            self.distribution=[self.get_coeff(n)**2 for n in range(-Walk.steps,Walk.steps+1)]
        return self.distribution
    
    def get_positions(self):
        """
        Returns list of all the positions between -Walk.steps and Walk.steps
        """
        return [n for n in range(-Walk.steps,Walk.steps+1)]
    
    def mean(self):
        """
        Calculates mean position

        Returns:
            mean
        """
        return sum([proba*position for proba,position in zip(self.get_distribution(),self.get_positions())])
    
    def variance(self):
        """
        Calculates variance of the positions
        
        Returns:
            variance"""

        e=self.mean()
        return sum([proba*(pos-e)**2 for proba,pos in zip(self.get_distribution(),self.get_positions())])
    
    def std(self):
        """
        Calculates standard deviation of the walk
        
        Returns:
            sqrt(variance())"""
        return np.sqrt(self.variance())
    
    def cardinal(self):
        """returns the number of position that are not of probability 0 divided by Walk.steps"""
        return len([proba for proba in self.get_distribution() if proba != 0])/Walk.steps
    
    def diversity(self):
        """returns the number of different values of the probability of the different positions divided by Walk.steps"""
        return len(list(set(self.get_distribution())))/Walk.steps
    
    def show(self):
        """let's show only even numbers (the odd numbers are of probability zero)"""
        print("WALK START :")
        print(f"Coin : {self.coin}")
        print(f"Particle : {'{:.3f}'.format(Walk.get_spin_coordinates(self.start)[0])}*|+> + {'{:.3f}'.format(Walk.get_spin_coordinates(self.start)[1])}*|->\n")
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        ax.set_xlabel("position")
        ax.set_ylabel("probabilité")
        ax.set_title(f"Marche à {Walk.steps} pas")
        plt.plot(
            np.arange(-Walk.steps,Walk.steps+1)[0::2],
            self.get_distribution()[0::2]
        )
        plt.show()
    

    @staticmethod
    def set_steps(steps:int):
        Walk.steps=steps
    
    @staticmethod
    def position(n:int):
        return fock(2*Walk.steps+1,Walk.steps+n)
    
    @staticmethod
    def spin(sign:str):
        dico = {
            "+":fock(2,0),
            "-":fock(2,1),
        }
        return dico[sign]
    
    @staticmethod
    def randomCoin(within_values:list=None):
        if within_values==None or len(within_values)==0:
            return np.random.rand(2,2)+1j*np.random.rand(2,2)
        else:
            return np.random.choice(within_values,size=(2,2))
    
    @staticmethod
    def randomParticle(within_values:list=None):
        if within_values==None or len(within_values)==0:
            up=np.random.rand()+1j*np.random.rand()
            down=np.random.rand()+1j*np.random.rand()
            return tensor((Walk.spin("+")*up+Walk.spin("-")*down),Walk.position(0))
        else:
            up=np.random.choice(within_values)
            down=np.random.choice(within_values)
            if up==down==0:
                up=1
            
            return tensor((Walk.spin("+")*up+Walk.spin("-")*down),Walk.position(0))
        
    @staticmethod
    def get_spin_coordinates(particle) -> tuple[float]:
        up = tensor(Walk.spin("+"),Walk.position(0)).dag()*particle
        down = tensor(Walk.spin("-"),Walk.position(0)).dag()*particle

        return up.full()[0][0],down.full()[0][0]
    

    
if __name__=="__main__":
    Walk.set_steps(100)

    coin=[
            [1,1],
            [1,-1]
        ]

    start=tensor(
        Walk.spin("+"),
        Walk.position(0)
    )

    simulation = Walk(start,coin)
    simulation.run(show_progress=True)