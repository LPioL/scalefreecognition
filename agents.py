from mesa import Agent
import random
from enum import IntEnum
import numpy as np
import sys
import multicellularity.schedule

sys.setrecursionlimit(10000)

limit = 1
multiplier = 20
stress_unity=2.5


class State(IntEnum):
    NEUTRAL = 0
    POLARIZED = 1
    DEPOLARIZED = 2


class Cell(Agent):
    """
    A cell that walks according to the outputs of an evolved neural network, and eat food to stay alive.
    """
    
    net=None
    depth = None
    count_food = 0
    count_toxines = 0
    grid = None
    x = None
    y = None
    energy = None
    energyt1=None
    state=None
    statet1=None
    moore = True   
    molecules = None
    goal=None
    opened_GJ = None
    opened_GJ_stress = None
    stress = None
    stresst1=None
    decision_state0 = None
    decision_state1= None
    decision_state2 = None


    def __init__(self, net, depth, unique_id, pos, model, moore, molecules, energy, energyt1,  goal, opened_GJ,opened_GJ_stress, stress, stresst1, decision_state0, decision_state1, decision_state2, state, statet1, state_tissue):
        """
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        moore: If True, may move in all 8 directions.
                Otherwise, only up, down, left, right.
        """
        super().__init__(unique_id, model)
        self.net=net
        self.depth=depth
        self.pos = pos
        self.moore = moore
        self.energy = energy
        self.energyt1 = energyt1
        self.state = state 
        self.statet1 = state 
        self.molecules = molecules
        self.goal = goal
        self.opened_GJ = opened_GJ
        self.opened_GJ_stress = opened_GJ_stress
        self.stress = stress
        self.stresst1 = stresst1
        self.state_tissue = state_tissue
        self.decision_state0 = decision_state0
        self.decision_state1 = decision_state1
        self.decision_state2 = decision_state2
        
        
    def net_input(self, tissue_matrix):

        # #INPUT network      
        new_input = []
        new_input.append(list(self.molecules))
        new_input = sum(new_input, [])
        new_input.append(self.energy)
        new_input.append(self.energyt1)
        new_input.append(self.stress)
        new_input.append(self.stresst1)
        new_input.append(self.state)
        new_input.append(self.statet1)  
        new_input.append(self.local_geometrical_frustration())  
        pos = list(self.pos)
        new_input.append(tissue_matrix[pos[0], pos[1]])   # how big the collective is
        new_input.append(np.abs(100 - (tissue_matrix[pos[0], pos[1]]/(self.model.height*self.model.width/3)*100)))


        new_input.append(0.5) # the bias 

        return(new_input)
    
    def net_output(self,new_input):
        
        #OUTPUT network    
        self.net.Flush()
        self.net.Input(new_input)
        [self.net.Activate() for _ in range(self.depth)]
        output = list(self.net.Output())  
        
        return output
    
    def update_state(self):
        
        if self.molecules[0] >=  10 :
            if self.state!=1:
                #self.energy -=0.5
                self.statet1 = self.state
                self.state = 1
            else: self.state = 1
           # else:
           #     self.energy -=0.25
           
        elif  self.molecules[0] < 10 and self.molecules[0] >= 5:
            if self.state!=3:
                #self.energy -=0.5
                self.statet1 = self.state
                self.state = 3
            else: self.state = 3

        elif self.molecules[0] >= 0 and self.molecules[0] < 5:
            if self.state!=2:
                #self.energy -=0.5
                self.statet1 = self.state
                self.state = 2    
            else: self.state = 2    
           # else:
           #     self.energy -=0.25
            
    def update_stress(self):
    
        if self.stress > 100:
            self.stress = 100
        if self.stress < 0:
            self.stress=0           
            
    def send_ions_and_stress(self, output):
  
        break_out_flag = False
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        j=0
        

        if self.pos == (0,0):
            positions = [[0,2],[1,2]] # positions neighbour
            GJ_directions = [2,3] 
        elif self.pos == (self.model.height-1,0):
            positions = [[0,1],[1,2]] # positions neighbour
            GJ_directions = [3,0] 
        elif self.pos==(0,self.model.height-1):
            positions = [[0,1],[1,2]] # positions neighbours 
            GJ_directions = [1,2]  # left up
        elif self.pos == (self.model.height-1, self.model.height-1):
            positions = [[0,1],[0,2]] # positions neighbour
            GJ_directions = [1,0]   
        elif self.pos[0]!=0 and self.pos[0]!=  self.model.height-1 and self.pos[1]==0: # milieu bas
            positions = [[0,1],[1,2,4], [3,4]] # positions neighbours
            GJ_directions = [1,2,3]     
        elif self.pos[0]==0 and self.pos[1]!=0 and self.pos[1]!=self.model.height-1: # mileu gauche
            positions = [[0,2],[2,3,4], [1,4]] # positions neighbours
            GJ_directions = [0,3,2]  
        elif self.pos[0]==self.model.height-1 and self.pos[1]!=0 and self.pos[1]!=self.model.height-1: #milieu droit
            positions = [[0,1,2],[0,3], [2,4]] # positions neighbours
            GJ_directions = [1,0,2]  
        elif self.pos[0]!=0 and self.pos[0]!= self.model.height-1 and self.pos[1]==self.model.height-1: #milieu haut
            positions = [[0,1],[0,2,3], [3,4]] # positions neighbours
            GJ_directions = [1,0,3]
        else:
            positions = [[2,4,7],[5,6,7],[0,3,5],[0,1,2]] # positions neighbours haut2 droite3 bas0 gauche1  2,3,0,1
            GJ_directions = [2,3,0,1]
            
        opposite_directions = {  "2": 0,"3": 1,"0": 2,"1": 3}


        for i in range(len(GJ_directions)):
            if self.opened_GJ[GJ_directions[i]]>0:
                
                for j in (positions[i]):
                    if self.model.grid.is_cell_empty(neighbours[j])==False: # not dead
                        cell_in_contact = self.model.grid[neighbours[j]][0]               
                        #print(cell_in_contact.pos)
                        
                        for k in range(len(self.molecules)):

                           if self.molecules[k]>=output[k]*((self.opened_GJ[GJ_directions[i]]*cell_in_contact.opened_GJ[opposite_directions[str(GJ_directions[i])]])):
                               cell_in_contact.molecules[k] += output[k]*((self.opened_GJ[GJ_directions[i]]*cell_in_contact.opened_GJ[opposite_directions[str(GJ_directions[i])]]))   
                               self.molecules[k] -= output[k]*((self.opened_GJ[GJ_directions[i]]*cell_in_contact.opened_GJ[opposite_directions[str(GJ_directions[i])]]))
                           else:
                               cell_in_contact.molecules[k] += self.molecules[k]
                               self.molecules[k] = 0
                        cell_in_contact.update_state()

                                       
        
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                 cell_in_contact = self.model.grid[neighbour][0]
               #  print(cell_in_contact.pos)
                
                 if self.opened_GJ_stress>0:
                     cell_in_contact.stress+=output[len(self.molecules)]*((self.opened_GJ_stress * cell_in_contact.opened_GJ_stress))   
                     cell_in_contact.stress-=output[2]*((self.opened_GJ_stress * cell_in_contact.opened_GJ_stress))   
                 cell_in_contact.update_stress()

        if self.opened_GJ_stress>0:
 
             self.stress+=output[len(self.molecules)] 
             self.stress-=output[2] 

            

                             
        for i in range(len(self.molecules)):
            if self.molecules[i]<0:
                self.molecules[i] = 0
        

        self.update_state()       


        
    def communication(self, tissue_matrix):
        """Find cell neighbours and pass information and/or energy. It 
           represents basic features of gap junctions."""

                    
        new_input = self.net_input( tissue_matrix)
        output  = self.net_output(new_input)
        
        for i in range(len(self.molecules)):
            if output[i]<0:
               output[i]=0 

        self.opened_GJ = [output[-1], output[len(output)-1], output[len(output)-2], output[len(output)-3]]
        self.opened_GJ_stress = output[len(output)-4]
        for i in range (len(self.opened_GJ)):
            if self.opened_GJ[i]>1:
                self.opened_GJ[i] = 1
            if self.opened_GJ[i]<0:
                self.opened_GJ[i] = 0   
        if self.opened_GJ_stress>1:
            self.opened_GJ_stress = 1
        if self.opened_GJ_stress<0:
            self.opened_GJ_stress = 0   
            


        self.send_ions_and_stress(output)
        
        return output
        
                
    def local_geometrical_frustration (self):
        geometrical_frustration = 0
        dead = 0
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                 cell_in_contact = self.model.grid.get_cell_list_contents([neighbour])[0]
                 if self.state != cell_in_contact.state:
                     geometrical_frustration += 1
            else:
                dead += 1

        return geometrical_frustration / (len(neighbours))
                     

                    
    def step(self, reward_mat, stress_mat, fitness_ff, tissue_matrix):
        """
        A model step. 
        """
        
        self.energyt1 = self.energy
        self.stresst1 = self.stress
        output = self.communication( tissue_matrix)
        

        reward = reward_mat[self.pos]
        stress = stress_mat[self.pos]

        self.energy += reward - 0.8



            
        if self.stress > 100:
            self.stress = 100
        if self.stress < 0:
            self.stress=0
            

        
        # Death
        if self.energy <= 0:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)
            
            
        