"""
Transition to multicellularity Model
================================

Léo Pio-Lopez
"""

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from multicellularity.schedule import RandomActivationByBreed
from multicellularity.agents import Cell
import numpy as np
import random        
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from pyinform import active_info, transfer_entropy
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import os 
import matplotlib.colors as c
import scipy.ndimage as ndimage
import matplotlib.ticker as mticker
from  matplotlib.ticker import FuncFormatter

limit = 1
multiplier = 20


class Multicellularity_model(Model):
    """
    Multicellularity Model
    """

    description = (
        "A model for simulating transition to multicellularity."
    )

    def __init__(
        self,
        net=None,
        depth=None,
        height= None, # for the vizualization grid
        width= None,  # idem
        initial_cells=None,
        initial_toxines=None,
        initial_food=None,
        cell_gain_from_food=None,
        cell_loss_from_toxine=None,
        energy = None,
        step_count = None,
        fitness =  None,
        nb_gap_junctions = None,
        cell_gain_from_good_state = None,
        goal = None,
        traj = [],
        traj1=[],
        states_matrix = [],
        states_timeseries=[],
        stress_matrix=[],
        GJ_matrix0  = [],
        GJ_matrix1  = [],
        GJ_matrix2  = [],
        GJ_matrix3  = [],
        energy_matrix  = [],
        digit_energy_matrix=[],
        molecules_matrix  = [],
        fitness_evaluation=False
    ):
        """
        Create a new multicellularity model with the given parameters.

        Args:
            initial_cell: Number of cell to start with
            initial_toxines: Number of toxines to start with
            cell_gain_from_food: Energy a cell gains from eating a food
            grass: Whether to have the sheep eat grass for energy
            grass_regrowth_time: How long it takes for a grass patch to regrow
                                 once it is eaten
        """
        super().__init__()
        # Set parameters
        self.net=net
        self.depth=depth
        self.height = height
        self.width = width
        self.initial_cells = initial_cells
        self.initial_toxines = initial_toxines
        self.initial_food = initial_food
        self.cell_gain_from_food = cell_gain_from_food
        self.cell_loss_from_toxine = cell_loss_from_toxine
        self.energy = energy
        self.step_count = step_count
        self.nb_gap_junctions = nb_gap_junctions
        self.fitness = fitness
        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=False)
        self.goal = goal
        self.traj = traj
        self.traj1 = traj1
        self.states_timeseries = states_timeseries
        self.cell_gain_from_good_state = cell_gain_from_good_state
        self.states_matrix = states_matrix
        self.stress_matrix = stress_matrix
        self.GJ_matrix0 = GJ_matrix0
        self.GJ_matrix1 = GJ_matrix1
        self.GJ_matrix2 = GJ_matrix2
        self.GJ_matrix3 = GJ_matrix3
        self.energy_matrix = energy_matrix
        self.digit_energy_matrix = digit_energy_matrix
        self.molecules_matrix = molecules_matrix
        self.datacollector = DataCollector(
            {


                "Internal stress": lambda m:m.schedule.get_internal_stress(Cell),


            }
        )


        # Create cells on the whole grid
        self.schedule.generate_cells(Cell, randomization=False)


            
        self.running = True
        
        if fitness_evaluation==False:
            self.datacollector.collect(self)




              

        
    def step(self, fitness_evaluation=True):
        
        reward_mat, stress_mat=self.schedule.reward_by_patches()
        fitness_ff=self.schedule.french_flag()
        tissue_matrix, state_matrix, stress_matrix, energy_matrix, molecules_matrix = self.schedule.adaptive_tissue()
        self.states_timeseries.append(state_matrix)
        self.states_matrix.append(state_matrix)
        self.stress_matrix.append(np.digitize(stress_matrix, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]))
        GJ_matrix0, GJ_matrix1, GJ_matrix2, GJ_matrix3 = self.schedule.get_GJ()
        self.GJ_matrix0.append(GJ_matrix0)
        self.GJ_matrix1.append(GJ_matrix1)
        self.GJ_matrix2.append(GJ_matrix2)
        self.GJ_matrix3.append(GJ_matrix3)
        self.energy_matrix.append(energy_matrix)
        self.digit_energy_matrix.append(np.digitize(energy_matrix, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]))
        #print('test')
        self.molecules_matrix.append(molecules_matrix)
        #print(stress_matrix)
        
        self.schedule.step(Cell, reward_mat, stress_mat, fitness_ff, tissue_matrix )
        self.schedule.update_state_tissue_costStateChange() #        self.schedule.update_state_tissue_costStateChange()

        
        if fitness_evaluation== False:
            mol=0
            for i in range(self.height):
               for j in range(self.width):
                   
                   if len(self.grid.get_cell_list_contents([(j,i)]))>0:
                       cell = self.grid.get_cell_list_contents([(j,i)])[0]  
                       mol+=cell.molecules[0]        
            
            #print(mol)
            #print('mol')

        # collect data
        self.datacollector.collect(self)
        

            
            
              
     

                
    def run_model(self, fitness_evaluation):

        # if self.verbose:
        #     print("Initial number cell: ", self.schedule.get_breed_count(Cell))
        
        for i in range(self.step_count):
            
            self.step(fitness_evaluation)

        if fitness_evaluation==True:
    
            self.fitness_ff=0
            general_energy=0
            bonus=0
            for i in range(self.height):
                 for j in range(int(self.width)):
                     if len(self.grid.get_cell_list_contents([(j,i)]))>0:
                         cell = self.grid.get_cell_list_contents([(j,i)])[0]
                         if cell.state_tissue == cell.goal:
                             self.fitness_ff+=1
                         general_energy+=cell.energy
                         if cell.energy>self.energy:
                             bonus+=1
                             
            self.fitness_ff = self.fitness_ff/(self.height*self.width)*100
            general_energy   = general_energy/ (self.height*self.width*10)        
            

                        
            remaining_cells = self.schedule.get_breed_count(Cell)
            if remaining_cells == 0:
                remaining_cells=1
    
            self.fitness = self.fitness_ff  

        return self.fitness                
        
                

            
 
  
