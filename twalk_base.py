from qutip import Qobj, fock ,mesolve, tensor
import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx

class tWalk:
    """tWalk represents a Walk of dimension tWalk.DIM (you shoud define it when inhertting !)
    You should also define <plot_ax(self,ax:Axes3D,time:int):> for plotting on an ax (ax3D if DIM>1)
    All the rest is fine !
    """
    
    DIM=None
    time_unit="u"
    x_label=""
    y_label=""
    z_label=""
    
    #INITIALIZE OBJECTS    
    def __init__(self,grid_size:int):
        assert self.defined(),"You haven't correcty defined the class, some static attributes are missing !"
        self.size=grid_size
        self.particle_over_time=None
        self.init_objects()
    
    def init_objects(self)->None:
        self.hamiltonian = tensor(*[
            Qobj(np.zeros((self.size,self.size))) for _ in range(self.DIM)
        ])
        self.particle = self.fock(*[self.size//2 for _ in range(self.DIM)])
    
    def __str__(self)->str:
        out=f"{type(self).__name__} :\n"
        for attr in ["size","frame_nbr","duration"]:
            out+=f"\t- {attr}:{getattr(self,attr,'None')}\n"
        out+="\t- H :\n"
        #let's print the hamiltonian (just the interesting relations, when it's not zero)
        for state1 in self.positions():
            for state2 in self.positions():
                coeff = self[state1+state2]
                if coeff==0:
                    continue
                if np.isreal(coeff):
                    coeff = coeff.real
                
                part1_str = " ".join([str(pos) for pos in state1])
                part2_str = ",".join([str(pos) for pos in state2])
                out+=f"\t\t- <{part1_str}|H|{part2_str}> = {coeff:.2e}\n"
        return out

    def __repr__(self)->str:
        out = f"Walk of type {type(self).__name__} :\n \
        \t - Dimension of the walk : {self.DIM}D\n\
        \t - Size of the grid : "+"x".join([str(self.size)]*self.DIM)+f"\n\
        \t - Space size : {self.part_dim}\n\
        \t - Hamiltonian size : {self.ham_dim}\n\
        \t - Walk solved : {self.particle_over_time!=None}\n\
        \t - Parameters initialized : {self.frame_nbr!=None and self.duration!=None}\n\
        "
        return out
    
    @property
    def part_dim(self):
        return self.size**self.DIM
    
    @property
    def ham_dim(self):
        return self.part_dim**2
    
    def fock(self,*args:int)->Qobj:
        """creates fock state as tensor(|arg_1>,...,|arg_n>)
        number of args should correspond to DIM
        """
        assert len(args)==self.DIM,f"Invalid number of arguments, there should be {type(self).DIM} and not {len(args)} !"
        focks = [
            fock(self.size,arg) for arg in args
        ]
        return tensor(*focks)
    
    def set_animation_param(self,duration:int=None,frame_nbr:int=None):
        """In the end, the animation will display the <frame_nbr> particles computed over time.
        DUration of the simulation is <duration>.
        Thus, time=np.linspace(0,duration,frame_nbr)"""
        if duration != None:
            self.duration = duration
        if frame_nbr!=None:
            self.frame_nbr=frame_nbr
        if self.duration==None or self.frame_nbr==None:
            print("WARNING, animation parameters have not been completely set !")
    
    #GET INFORMATION
    def to_index(self,*args)->int:
        """translates coordinates to the corresponding index of the vector after tensor product has been applied"""
        assert len(args)==self.DIM,f"IndexError, there should be {self.DIM} positionnal arguments !"
        out=0
        for pos in args:
            out*=self.size
            out+=pos
        return out
    
    def part_coeff(self,*args:int,time:int=None)->complex:
        """returns coefficient of the state tensor(|arg_1>,...,|arg_n>
        if t!=None, it corresponds to the particle at time t        
        """
        if time==None:
            part=self.particle
        else:
            part=self.particle_over_time[time]
        return part[self.to_index(*args)][0][0]
    
    def get_distribution(self,time:int)->np.array:
        """computes and returns the distribution of probabilities

        Args:
            time (int): wich particle are we looking at ?

        Returns:
            np.array: P(pos=x,y,z)= distribution[x,y,z]
        """
        dist = np.zeros([self.size for _ in range(self.DIM)])
        for state in self.positions():
            coeff = self.part_coeff(*state,time=time)
            dist[state]=abs(coeff)**2
        return dist
    
    def mean(self,time:int)->np.ndarray:
        dist = self.get_distribution(time=time)
        return sum(dist[pos]*np.array(pos) for pos in self.positions())
    
    def variance(self,time:int)->float:
        e = self.mean(time)
        dist = self.get_distribution(time)
        
        def distance(pos1:np.ndarray,pos2:np.ndarray):
            return np.linalg.norm(pos1-pos2)**2
        
        return sum(distance(np.array(pos),e)*dist[pos] for pos in self.positions())
    
    def std(self,time:int)->float:
        return np.sqrt(self.variance(time))
    
    def plot_ax(self,ax:Axes3D,time:int):
        raise(Exception(f"Undefined method {type(self).__name__}.plot_ax !"))
    
    
        
        
    #ACT ON HAMILTONIAN
    def pack(self)->None:
        """corrects hamiltonian to make it unitary and hermitian"""
        self.hamiltonian+=self.hamiltonian.dag()
        self.hamiltonian=self.hamiltonian.unit()
    
    def __setitem__(self,coords:tuple,coeff:complex)->None:
        coords=list(coords)
        pos1 = coords[:self.DIM]
        pos2 = coords[self.DIM:]
        x1,x2 = self.to_index(*pos1),self.to_index(*pos2)
        
        arr = self.hamiltonian.data.toarray()
        arr[x2,x1]=coeff
        self.hamiltonian = Qobj(arr) #HORRIBLE !!! TROUVERUNE SOLUTION
    
    def __getitem__(self,coords:tuple) -> complex:
        coords=list(coords)
        pos1 = coords[:self.DIM]
        pos2 = coords[self.DIM:]
        pos1 = self.to_index(*pos1)
        pos2 = self.to_index(*pos2)
        return self.hamiltonian[pos2,pos1]
    
    def set_hamiltonian(self,hamiltonian:np.ndarray):
        """It is more efficient to first set the hamiltonian as a numpy array and then inject it in the walk,
        because in place modification of a Qobj is (to my knowledge) impossible"""
        self.hamiltonian=Qobj(hamiltonian)
    
    #RUN ANIMATION
    def solve(self)->None:
        """runs the simulation after calling self.pack()"""
        self.pack()
        assert self.duration!=None,"Please use set_animation_param(duration=...)"
        assert self.frame_nbr!=None,"Please use set_animation_param(frame_nbr=...)"
        if self.particle_over_time!=None:
            return
        time=np.linspace(0,self.duration,self.frame_nbr)
        self.particle_over_time = mesolve(self.hamiltonian,self.particle,time).states
    
    def run(self,keep_scale:bool=False)->None:
        if self.particle_over_time==None:
            self.solve()
        
        fig = plt.figure()
        if self.DIM>1:
            ax = fig.add_subplot(1,1,1,projection='3d')
        else:
            ax = fig.add_subplot(1,1,1)
        
        def animate(t:int):
            if t>=self.frame_nbr:
                return
            
            print(f"Mean = {self.mean(t)} | STD = {self.std(t):.2f}"+" "*20,end="\r")
            fig.suptitle(f"Time : {min(t,self.frame_nbr-1)+1} {self.time_unit}")
            ax.clear()
            
            if keep_scale:
                if self.DIM==1:
                    ax.set_ylim(0,1)
                if self.DIM==2:
                    ax.set_zlim(0,1)
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self.y_label)
            if self.DIM>=2:
                ax.set_zlabel(self.z_label)
            
            self.plot_ax(ax,t)
        
        ani = anim.FuncAnimation(fig,animate,interval=10)
        plt.show()
    
    #ITERATORS
    
    def positions(self):
        """returns all the grid positions"""
        ranges=[(i for i in range(self.size)) for _ in range(self.DIM)]
        return itertools.product(*ranges)
    
    def neighbours(self,*coords:int):
        """returns all the neighbours of one point of the grid"""
        assert len(coords)==self.DIM,f"Invalid number of arguments, there should be {type(self).DIM} and not {len(coords)} !"
        
        def is_ok(new_coords:tuple):
            for x in new_coords:
                if x<0 or x>=self.size:
                    return False
            return True
        
        out=[]
        for i in range(self.DIM):
            move_up = [1 if j==i else 0 for j in range(self.DIM)]
            move_down = [-move for move in move_up]
            
            point=[]
            for k in range(len(coords)):
                new = coords[k]+move_up[k]
                point.append(new)
            if is_ok(point):
                out.append(point)
            
            point=[]
            for k in range(len(coords)):
                new = coords[k]+move_down[k]
                point.append(new)
            if is_ok(point):
                out.append(point)
                
        return out
        
    @classmethod
    def defined(cls:type)->bool:
        attrs_ok = [
            cls.DIM!=None,
        ]
        return all(attrs_ok)
    
    @classmethod
    def uniform_tunneling(cls:type,grid_size:int):
        walk:tWalk = cls(grid_size)
        walk.particle=walk.fock(*[grid_size//2 for _ in range(walk.DIM)])
        hamiltonian = np.zeros((walk.part_dim,walk.part_dim))
        
        for state in walk.positions():
            for neighbour in walk.neighbours(*state):
                state_index = walk.to_index(*state)
                neighbour_index = walk.to_index(*neighbour)
                hamiltonian[state_index][neighbour_index]=1
        
        walk.set_hamiltonian(hamiltonian)
        return walk
    
    @classmethod
    def random_tunneling(cls:type,grid_size:int):
        walk:tWalk = cls(grid_size)
        walk.particle=walk.fock(*[grid_size//2 for _ in range(walk.DIM)])
        hamiltonian = np.zeros((walk.part_dim,walk.part_dim))
        
        for state in walk.positions():
            for neighbour in walk.neighbours(*state):
                state_index = walk.to_index(*state)
                neighbour_index = walk.to_index(*neighbour)
                hamiltonian[state_index][neighbour_index]=np.random.rand()
        
        walk.set_hamiltonian(hamiltonian)
        return walk


if __name__ == "__main__":
    tWalk.DIM=3
    walk = tWalk(5)
    print(walk.particle)
    print(walk.part_coeff(2,2,2))
    walk[1,2,3,3,2,1]=1
    walk.pack()
    print(walk)
    
    print(walk.neighbours(0,0,1))
    
    walk = tWalk.uniform_tunneling(5)
    print(walk.get_distribution(None))
    print(walk.mean(None))
    print(walk.std(None))