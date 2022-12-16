from collections import defaultdict
from mesa.time import RandomActivation
import numpy as np
from spatialentropy import leibovici_entropy, altieri_entropy
import random 
from scipy.ndimage.measurements import label


class RandomActivationByBreed(RandomActivation):
    """
    A scheduler which activates each type of agent once per step, in random
    order, with the order reshuffled every step.

    This is equivalent to the NetLogo 'ask breed...' and is generally the
    default behavior for an ABM.

    Assumes that all agents have a step() method.
    """

    def __init__(self, model):
        super().__init__(model)
        self.agents_by_breed = defaultdict(dict)

    def add(self, agent):
        """
        Add an Agent object to the schedule

        Args:
            agent: An Agent to be added to the schedule.
        """

        self._agents[agent.unique_id] = agent
        agent_class = type(agent)
        self.agents_by_breed[agent_class][agent.unique_id] = agent

    def remove(self, agent):
        """
        Remove all instances of a given agent from the schedule.
        """

        del self._agents[agent.unique_id]

        agent_class = type(agent)
        del self.agents_by_breed[agent_class][agent.unique_id]

    def step(self, Cell, reward_mat, stress_mat, fitness_ff, tissue_matrix,  by_breed=True):
        """
        Executes the step of each agent breed, one at a time, in random order.

        Args:
            by_breed: If True, run all agents of a single breed before running
                      the next one.
        """

        if by_breed:
            for agent_class in self.agents_by_breed:
                self.step_breed(Cell, agent_class, reward_mat, stress_mat, fitness_ff, tissue_matrix)
            self.steps += 1
            self.time += 1
        else:
            super().step()

    def step_breed(self, Cell, breed, reward_mat, stress_mat, fitness_ff, tissue_matrix ):
        """
        Shuffle order and run all agents of a given breed.

        Args:
            breed: Class object of the breed to run.
        """

        agent_keys = list(self.agents_by_breed[breed].keys())
        self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            self.agents_by_breed[breed][agent_key].step(reward_mat, stress_mat, fitness_ff, tissue_matrix)



    def get_breed_count(self, breed_class):
        """
        Returns the current number of agents of certain breed in the queue.
        """
        return len(self.agents_by_breed[breed_class].values())
    


    def get_internal_stress(self, Cell):
        
        stress = 0
        for i in range(self.model.height):
            for j in range(self.model.width):
                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell= self.model.grid.get_cell_list_contents([(j,i)])[0]                        
                    stress+=cell.stress
        return stress
        

    

    def adaptive_tissue(self):
        
       openedGJ_matrix = np.zeros ((self.model.height, self.model.width))        
       state_matrix = np.zeros ((self.model.height, self.model.width))
       stress_matrix = np.zeros ((self.model.height, self.model.width))
       tissue_matrix =  np.zeros ((self.model.height, self.model.width))
       energy_matrix =  np.zeros ((self.model.height, self.model.width))
       molecules_matrix =  np.zeros ((self.model.height, self.model.width))

       for i in range(self.model.height):
           for j in range(int(self.model.width)):
              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                  if sum(cell.opened_GJ)>0:
                      openedGJ_matrix[j,i]=1    
                  state_matrix[j,i] = cell.state_tissue
                  stress_matrix[j,i] = cell.stress
                  energy_matrix[j,i] = cell.energy
                  molecules_matrix[j,i] = cell.molecules[0]

    
       state_matrix1 = np.where((state_matrix!=1), 0, state_matrix)
       state_matrix2 = np.where((state_matrix!=2), 0, state_matrix)
       state_matrix3 = np.where((state_matrix!=3), 0, state_matrix)

       state_matrices = [state_matrix1, state_matrix2, state_matrix3]

       structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter


       for j in range(len(state_matrices)):
           labeled, ncomponents = label(np.array(state_matrices[j]), structure)
           indices = np.indices(state_matrices[j].shape).T[:,:,[1, 0]]
           if ncomponents > 0:
               for i in range (ncomponents):
                   positions = indices[labeled == i+1]
                   for k in range(len(positions)):
                       tissue_matrix[positions[k][0], positions[k][1] ] = len(positions)

       
       return tissue_matrix, state_matrix, stress_matrix, energy_matrix, molecules_matrix
            
                    
    def french_flag(self):
        self.fitness_ff=0
        for i in range(self.model.height):
             for j in range(int(self.model.width)):
                 if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                     cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                     if cell.state == cell.goal:
                         self.fitness_ff+=1
                         
        
        self.fitness_ff = self.fitness_ff/(self.model.height*self.model.width)*100
        
        return self.fitness_ff
    

    def get_cellular_stress(self, Cell):
        
        stress = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())
        for agent_key in agent_keys:
                stress+=self.agents_by_breed[Cell][agent_key].stress
        return stress
        
    def get_GJ(self):
        
        GJ_matrix0 = np.zeros ((self.model.height, self.model.width))     
        GJ_matrix1 = np.zeros ((self.model.height, self.model.width))        
        GJ_matrix2 = np.zeros ((self.model.height, self.model.width))        
        GJ_matrix3 = np.zeros ((self.model.height, self.model.width)) 
 
        for i in range(self.model.height):
            for j in range(int(self.model.width)):
                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0: 
                    cell = self.model.grid.get_cell_list_contents([(j,i)])[0]    
                    GJ_matrix0[i][j] =  cell.opened_GJ[0]
                    GJ_matrix1[i][j] =  cell.opened_GJ[1]
                    GJ_matrix2[i][j] =  cell.opened_GJ[2]
                    GJ_matrix3[i][j] =  cell.opened_GJ[3]
        return GJ_matrix0, GJ_matrix1, GJ_matrix2, GJ_matrix3  
    
                        
    def update_state_tissue_costStateChange(self):
        for i in range(self.model.height):
             for j in range( self.model.width):
                 if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                    if cell.molecules[0] >= 10 :
                        if cell.state_tissue == 1:
                            cell.state_tissue = 1
                        else:
                            cell.state_tissue = 1
                            cell.energy -=0.25
            
            
                    elif cell.molecules[0] < 10 and cell.molecules[0] >= 5:
                        if cell.state_tissue == 3:
                            cell.state_tissue = 3
                        else:
                            cell.state_tissue = 3
                            cell.energy -=0.25
                        
                    elif cell.molecules[0] >= 0 and cell.molecules[0] < 5:
                        if cell.state_tissue == 2:
                            cell.state_tissue = 2
                        else:
                            cell.state_tissue = 2
                            cell.energy -=0.25
                         
    def reward_by_patches(self):
        
        reward_mat=np.zeros((self.model.width, self.model.height))
        stress_mat=np.zeros((self.model.width, self.model.height))
        
        reward=0
        stress = 0
        not_dead=1
        stripe1 = []
        for i in range(self.model.height):
            for j in range(int(self.model.width/3)):
                 if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                     not_dead+=1
                     cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                     if cell.state_tissue == cell.goal:
                         reward+=1
                         stripe1.append(cell.pos)
                     else:
                         stress+=1
        for i in range(self.model.height):
            for j in range(int(self.model.width/3)):
                reward_mat[j,i] = reward/not_dead
                stress_mat[j,i] = stress/not_dead


        reward=0
        stress = 0
        not_dead=1
        stripe2 = []
        for i in range(self.model.height):
             for j in range(int(self.model.width/3), int(2*self.model.width/3)):
                 if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                     not_dead+=1
                     cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                     if cell.state_tissue == cell.goal:
                         reward+=1
                         stripe2.append(cell.pos)
                     else:
                         stress+=1
        for i in range(self.model.height):
             for j in range(int(self.model.width/3), int(2*self.model.width/3)):            
                 reward_mat[j,i] = reward/not_dead
                 stress_mat[j,i] = stress/not_dead


        reward=0
        stress = 0
        not_dead=1
        stripe3 = []
        for i in range(self.model.height):
             for j in range(int(2*self.model.width/3), self.model.width):
                 if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                     not_dead+=1
                     cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                     if cell.state_tissue == cell.goal:
                         reward+=1
                         stripe3.append(cell.pos)

                     else:
                         stress+=1
        for i in range(self.model.height):
            for j in range(int(2*self.model.width/3), self.model.width):
                reward_mat[j,i] = reward/not_dead
                stress_mat[j,i] = stress/not_dead
                
        return reward_mat, stress_mat

    
    def generate_cells(self, Cell, randomization=True):
        

         if randomization == True:
         # Create cells on the whole grid
             mat1= [[ 3, 3, 3, 2, 2, 2, 1, 1, 1],
                     [ 3, 3, 3, 2, 2, 2, 1, 1, 1],
                     [ 3, 3, 3, 2, 2, 2, 1, 1, 1],
                     [ 3, 3, 3, 2, 2, 2, 1, 1, 1],
                     [ 3, 3, 3, 2, 2, 2, 1, 1, 1],
                     [ 3, 3, 3, 2, 2, 2, 1, 1, 1],
                     [ 3, 3, 3, 2, 2, 2, 1, 1, 1],
                     [ 3, 3, 3, 2, 2, 2, 1, 1, 1],
                     [ 3, 3, 3, 2, 2, 2, 1, 1, 1]]
             
             mat = [[3, 1, 2, 2, 1, 2, 2, 3, 3],
                    [3, 2, 3, 1, 1, 2, 1, 3, 2],
                    [1, 3, 1, 2, 3, 1, 1, 2, 1],
                    [1, 3, 3, 1, 1, 2, 1, 3, 1],
                    [2, 1, 3, 2, 3, 3, 2, 1, 3],
                    [2, 3, 1, 2, 2, 2, 3, 1, 1],
                    [3, 3, 1, 3, 2, 2, 1, 1, 1],
                    [3, 1, 1, 1, 2, 3, 1, 1, 1],
                    [3, 2, 1, 1, 1, 3, 2, 2, 1]]
             
             mat= list(np.random.randint(3, size=(9, 9))+1)
             mat=[l.tolist() for l in mat]
             mat=mat1
             
             for i in range(self.model.height):
                 for j in range(self.model.width):
                     state_cell = mat[i][j]
                     state_tissue = mat[i][j]
                     molecules =  [random.random()] 
                     if mat[i][j] == 1:
                         molecules[0] =  11
                     elif mat[i][j] == 2:
                         molecules[0] = 3
                     elif mat[i][j] == 3:
                         molecules[0] = 7
                     cell = Cell(self.model.net, self.model.depth, self.model.next_id(), (j, i), self.model,  True, 
                                 energy = self.model.energy, energyt1 =  self.model.energy,  
                                 molecules = molecules, goal = self.model.goal[i][j], opened_GJ = [0,0,0,0], 
                                 opened_GJ_stress=0, stress = 0, stresst1=0, decision_state0=0, decision_state1=0,decision_state2=0, 
                                 state=state_cell, statet1=state_cell, state_tissue = state_tissue)
                     self.model.grid.place_agent(cell, (j, i))
                     self.model.schedule.add(cell)

         else:
             for i in range(self.model.height):
                 for j in range(self.model.width):
                     state_cell = 1
                     state_tissue = 1
                     molecules =  [random.random()] 
                     if state_cell == 1:
                         molecules[0] = 11
                     elif state_cell == 2:
                         molecules[0] = 3
                     elif state_cell == 3:
                         molecules[0] = 7
                     cell = Cell(self.model.net, self.model.depth, self.model.next_id(), (j, i), self.model,  True, 
                                 energy = self.model.energy, energyt1 =  self.model.energy,   
                                 molecules = molecules, goal = self.model.goal[i][j], opened_GJ = [0,0,0,0], 
                                 opened_GJ_stress=0, stress = 0, stresst1=0, decision_state0=0, decision_state1=0,decision_state2=0, 
                                 state=state_cell, statet1=state_cell, state_tissue = state_tissue)
                     self.model.grid.place_agent(cell, (j, i))
                     self.model.schedule.add(cell)
                     
                     
    
                     
                     
                     