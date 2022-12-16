from multicellularity.model_for_analysis import Multicellularity_model
try:
   import cPickle as pickle
except:
   import pickle
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from multicellularity.agents import Cell
import os
import sys
import time
import random as rnd
import subprocess as comm
import cv2
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
from MultiNEAT import GetGenomeList, ZipFitness
from MultiNEAT import  EvaluateGenomeList_Parallel, EvaluateGenomeList_Serial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multicellularity.visualize as visualize
import sys

from psutil import process_iter
from signal import SIGKILL

sys.setrecursionlimit(100000)

# Params for MESA model and the evolving neural network
config_dict = {}
f = open("general_params.txt")

for lines in f:
	lines
	items = lines.split(': ', 1)
	config_dict[items[0]] = eval(items[1])
fitness_function = "" # see model class for the definitions of the variable

nb_gens=config_dict["nb_gens"]
depth=config_dict["depth"]
height=config_dict["height"]
width=config_dict["width"]
max_fitness=config_dict["max_fitness"]
energy = config_dict["energy"]
nb_gap_junctions = config_dict["nb_gap_junctions"]
step_count = config_dict["step_count"]
fitness_function = "" # see model class for the definitions of the variable

# French flag


# French flag
goal = [[ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2]]




def agents_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.state ==4:
        portrayal["Color"] = ["yellow"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.state ==3:
        portrayal["Color"] = ["grey"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.state ==1:
        portrayal["Color"] = ["blue"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.state ==2:
        portrayal["Color"] = ["red"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        

    return portrayal

def agents_portraya_state_tissue(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.state_tissue ==4:
        portrayal["Color"] = ["yellow"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.state_tissue ==3:
        portrayal["Color"] = ["grey"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.state_tissue ==1:
        portrayal["Color"] = ["blue"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.state_tissue ==2:
        portrayal["Color"] = ["red"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        

    return portrayal

def agents_portrayal1(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and max(agent.opened_GJ)==agent.opened_GJ[0]:
        portrayal["Color"] = ["#173806"]
        portrayal["Shape"] = "arrowHead"
        portrayal["scale"] = 0.6
        portrayal["heading_x"] = 0
        portrayal["heading_y"] = 1
        portrayal["Layer"] = 0
     
        
    if type(agent) is Cell and max(agent.opened_GJ)==agent.opened_GJ[1]:
        portrayal["Color"] = ["#173806"]
        portrayal["Shape"] = "arrowHead"
        portrayal["scale"] = 0.6
        portrayal["heading_x"] = 1
        portrayal["heading_y"] = 0      
        portrayal["Layer"] = 0
      

    if type(agent) is Cell and max(agent.opened_GJ)==agent.opened_GJ[2]:
        portrayal["Color"] = ["#173806"]
        portrayal["Shape"] = "arrowHead"
        portrayal["scale"] = 0.6
        portrayal["heading_x"] = 0
        portrayal["heading_y"] = -1       
        portrayal["Layer"] = 0
      

    if type(agent) is Cell and max(agent.opened_GJ)==agent.opened_GJ[3]:
        portrayal["Color"] = ["#173806"]
        portrayal["Shape"] = "arrowHead"
        portrayal["scale"] = 0.6
        portrayal["heading_x"] = -1
        portrayal["heading_y"] = 0
        portrayal["Layer"] = 0
        
      
    return portrayal

def agents_portrayal2(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.stress ==0.0:
        portrayal["Color"] = ["#f9ebea"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.stress <=10  and agent.stress >0:
        portrayal["Color"] = ["#f2d7d5"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=20  and agent.stress >10:
        portrayal["Color"] = ["#e6b0aa"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=30 and agent.stress >20:
        portrayal["Color"] = ["#d98880"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    if type(agent) is Cell and agent.stress <=40  and agent.stress >30:
        portrayal["Color"] = ["#cd6155"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=50  and agent.stress >40:
        portrayal["Color"] = ["#c0392b"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1   
        
    if type(agent) is Cell and agent.stress <=60  and agent.stress >50:
        portrayal["Color"] = ["#a93226"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=70 and agent.stress >60:
        portrayal["Color"] = ["#922b21"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1  
    
    if type(agent) is Cell and agent.stress <=80  and agent.stress >70:
        portrayal["Color"] = ["#7b241c"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=90 and agent.stress >80:
        portrayal["Color"] = ["#641e16"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    if type(agent) is Cell and agent.stress >90:
        portrayal["Color"] = ["#1b2631"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    return portrayal

def agents_portrayal3(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        if type(agent) is Cell and agent.opened_GJ[0] ==0:
            portrayal["Color"] = ["grey"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1
            
        if type(agent) is Cell and agent.opened_GJ[0] <=0.25  and agent.opened_GJ[0] >0:
            portrayal["Color"] = ["#173806"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.opened_GJ[0] <=0.5  and agent.opened_GJ[0] >0.25:
            portrayal["Color"] = ["#2d6e0c"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.opened_GJ[0] <=0.75  and agent.opened_GJ[0] >0.5:
            portrayal["Color"] = ["#47b012"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
            
        if type(agent) is Cell  and agent.opened_GJ[0] >0.75:
            portrayal["Color"] = ["#62f716"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
        
        return portrayal
    
def agents_portrayal4(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        if type(agent) is Cell and agent.opened_GJ[1] ==0:
            portrayal["Color"] = ["grey"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1
            
        if type(agent) is Cell and agent.opened_GJ[1] <=0.25  and agent.opened_GJ[1] >0:
            portrayal["Color"] = ["#173806"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.opened_GJ[1] <=0.5  and agent.opened_GJ[1] >0.25:
            portrayal["Color"] = ["#2d6e0c"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.opened_GJ[1] <=0.75  and agent.opened_GJ[1] >0.5:
            portrayal["Color"] = ["#47b012"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
            
        if type(agent) is Cell  and agent.opened_GJ[1] >0.75:
            portrayal["Color"] = ["#62f716"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
        
        return portrayal
    
def agents_portrayal5(agent):
         if agent is None:
             return
     
         portrayal = {}
     
         if type(agent) is Cell and agent.opened_GJ[2] ==0:
             portrayal["Color"] = ["grey"]
             portrayal["Shape"] = "circle"
             portrayal["Filled"] = "true"
             portrayal["Layer"] = 0
             portrayal["r"] = 1
             
         if type(agent) is Cell and agent.opened_GJ[2] <=0.25  and agent.opened_GJ[2] >0:
             portrayal["Color"] = ["#173806"]
             portrayal["Shape"] = "circle"
             portrayal["Filled"] = "true"
             portrayal["Layer"] = 0
             portrayal["r"] = 1    
     
         if type(agent) is Cell and agent.opened_GJ[2] <=0.5  and agent.opened_GJ[2] >0.25:
             portrayal["Color"] = ["#2d6e0c"]
             portrayal["Shape"] = "circle"
             portrayal["Filled"] = "true"
             portrayal["Layer"] = 0
             portrayal["r"] = 1    
     
         if type(agent) is Cell and agent.opened_GJ[2] <=0.75  and agent.opened_GJ[2] >0.5:
             portrayal["Color"] = ["#47b012"]
             portrayal["Shape"] = "circle"
             portrayal["Filled"] = "true"
             portrayal["Layer"] = 0
             portrayal["r"] = 1    
             
         if type(agent) is Cell  and agent.opened_GJ[2] >0.75:
             portrayal["Color"] = ["#62f716"]
             portrayal["Shape"] = "circle"
             portrayal["Filled"] = "true"
             portrayal["Layer"] = 0
             portrayal["r"] = 1    
         
         return portrayal  
     
def agents_portrayal6(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        if type(agent) is Cell and agent.opened_GJ[3] ==0:
            portrayal["Color"] = ["grey"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1
            
        if type(agent) is Cell and agent.opened_GJ[3] <=0.25  and agent.opened_GJ[3] >0:
            portrayal["Color"] = ["#173806"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.opened_GJ[3] <=0.5  and agent.opened_GJ[3] >0.25:
            portrayal["Color"] = ["#2d6e0c"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.opened_GJ[3] <=0.75  and agent.opened_GJ[3] >0.5:
            portrayal["Color"] = ["#47b012"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
            
        if type(agent) is Cell  and agent.opened_GJ[3] >0.75:
            portrayal["Color"] = ["#62f716"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
        
        return portrayal

def agents_portrayal7(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        if type(agent) is Cell :
            portrayal["Color"] = ["yellow"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["text"] = str(int(agent.molecules[0]))
            portrayal["text_color"] = ["black"]
            portrayal["Layer"] = 0
            portrayal["r"] = 0.1

        
        return portrayal
    

    
canvas_element = CanvasGrid(agents_portrayal, height, width, 200, 200)
canvas_element1 = CanvasGrid(agents_portrayal1, height, width, 200, 200)
canvas_element2 = CanvasGrid(agents_portrayal2, height, width, 200, 200)
canvas_element3 = CanvasGrid(agents_portrayal3, height, width, 200, 200)
canvas_element4 = CanvasGrid(agents_portrayal4, height, width, 200, 200)
canvas_element5 = CanvasGrid(agents_portrayal5, height, width, 200, 200)
canvas_element6 = CanvasGrid(agents_portrayal6, height, width, 200, 200)
canvas_element7 = CanvasGrid(agents_portrayal7, height, width, 200, 200)
canvas_element9 = CanvasGrid(agents_portraya_state_tissue, height, width, 200, 200)






chart_element5 = ChartModule(
        [{"Label": "Internal stress", "Color": "black"}]
   # [{"Label": "Global stress", "Color": "#AA0000"}], [{"Label": "Global", "Color": "##84e184"}]
)

# If run as script.
if __name__ == '__main__':
    

# 
    seed = int(np.load("seed.npy"))
    rng = NEAT.RNG()
    rng.Seed(seed)
#    

    
    winner_net = NEAT.NeuralNetwork()
    winner_net.Load("winner_net.txt")
    winner_net.Flush()
    print("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))


    # Test the winner network on the server
    model_params = {
        "net":winner_net,
        "depth":depth-1,
        "height": height, # for the vizualization grid
        "width": width,  # idem
        "energy": energy,
        "step_count": step_count,
        "nb_gap_junctions":nb_gap_junctions,
        "fitness": fitness_function,
        "goal":goal,
        "fitness_evaluation": False
    }
            


    import socketserver
    
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
    
    server = ModularServer(
        Multicellularity_model, [canvas_element, canvas_element9, canvas_element2, canvas_element1, canvas_element3,canvas_element4,canvas_element5,canvas_element6, canvas_element7, chart_element5], "Multi-cellularity", model_params)
    server.port = free_port

    server.launch()
