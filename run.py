## 4 directions


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
import datetime
import shutil
import multiprocessing

cpu_number = multiprocessing.cpu_count()
print(cpu_number)




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
          
        

# Params for MESA model and the evolving neural network
nb_gens=2
depth=4
height=len(goal) # for the vizualization grid
width=len(goal)
max_fitness= 97
initial_cells=height*width
energy = 70
step_count = 100
fitness_function = "" # see model class for the definitions of the variable
nb_gap_junctions = 4
nb_stress_GJ = 1
nb_output_molecules = 1
nb_output_stress = 1

  
# Interface
interface =  False
# Params es-hyperneat
params = NEAT.Parameters()
params.PopulationSize = 350

params.DynamicCompatibility = True
params.CompatTreshold = 3.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 10
params.OldAgeTreshold = 35
params.StagnationDelta = 5
params.MinSpecies = 5
params.MaxSpecies = 15
params.RouletteWheelSelection = False

params.MutateRemLinkProb = 0.02
params.RecurrentProb = 0.2
params.OverallMutationRate = 0.15
params.MutateAddLinkProb = 0.03
params.MutateAddNeuronProb = 0.03
params.MutateWeightsProb = 0.90
params.MaxWeight = 8.0
params.MinWeight = -8.0
params.WeightMutationMaxPower = 0.2
params.WeightReplacementMaxPower = 1.0

params.MutateActivationAProb = 0.0
params.ActivationAMutationMaxPower = 0.5
params.MinActivationA = 0.05
params.MaxActivationA = 6.9 

params.MinNeuronBias = -params.MaxWeight
params.MaxNeuronBias = params.MaxWeight
    
params.MutateNeuronActivationTypeProb = 0.3
params.ActivationFunction_SignedGauss_Prob = 1.0
params.ActivationFunction_SignedStep_Prob = 1.0
params.ActivationFunction_Linear_Prob = 1.0
params.ActivationFunction_SignedSine_Prob = 1.0
params.ActivationFunction_SignedSigmoid_Prob = 1.0

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_TanhCubic_Prob = 0.0
params.ActivationFunction_UnsignedStep_Prob = 0.0
params.ActivationFunction_UnsignedGauss_Prob = 0.0
params.ActivationFunction_Abs_Prob = 0.0
params.ActivationFunction_UnsignedSine_Prob = 0.0

params.AllowLoops = True
params.AllowClones = True

params.MutateNeuronTraitsProb = 0
params.MutateLinkTraitsProb = 0


params.DivisionThreshold = 0.03 
params.VarianceThreshold = 0.03
params.BandThreshold = 0.3
# depth of the quadtree
params.InitialDepth = 3
params.MaxDepth = 3
# corresponds to the number of hidden layers = iterationlevel+1
params.IterationLevel = depth -1
params.Leo = False
params.GeometrySeed = True
params.LeoSeed = True
params.LeoThreshold = 0.2
params.CPPN_Bias = -1.0
params.Qtree_X = 0.0
params.Qtree_Y = 0.0
params.Width = 1.
params.Height = 1.
params.Elitism = 0.1

rng = NEAT.RNG()
rng.TimeSeed()



# Network inputs and expected outputs.
nb_inputs = 10 # nb molecules + energy + stress + state + bias + state_neigbours
nb_outputs =  nb_gap_junctions + nb_output_molecules + nb_output_stress + nb_stress_GJ+1#   molecules to send + stress to send + opened gap junction including stress GJ   + decision state change
input_coordinates = []
output_coordinates = []
for i in range(0, nb_inputs):
    input_coordinates.append((-1. +(2.*i/(nb_inputs - 1)), -1.))
for i in range(0, nb_outputs):
    output_coordinates.append((-1. +(2.*i/(nb_outputs - 1)),1.))

#  Substrate for MultiNEAT
substrate = NEAT.Substrate(input_coordinates, [],  output_coordinates)
substrate.m_allow_input_hidden_links = True
substrate.m_allow_hidden_output_links = True
substrate.m_allow_hidden_hidden_links = False
substrate.m_allow_looped_hidden_links = True

substrate.m_allow_input_output_links = False

substrate.m_allow_output_hidden_links = True
substrate.m_allow_output_output_links = False
substrate.m_allow_looped_output_links = False


substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

substrate.m_with_distance = True

substrate.m_max_weight_and_bias = 8.0

try:
    x = pickle.dumps(substrate)
except:
    print('You have mistyped a substrate member name upon setup. Please fix it.')
    sys.exit(1)



def eval_individual(genome):
    
    """
    Evaluate fitness of the individual CPPN genome by creating
    the substrate with topology based on the CPPN output.
    Arguments:
        genome:         The CPPN genome
        substrate:      The substrate to build control ANN
        params:         The ES-HyperNEAT hyper-parameters
    Returns:
        fitness_indiidual The fitness of the individual
    """
    
    net = NEAT.NeuralNetwork()
    genome.BuildESHyperNEATPhenotype(net, substrate, params)
    depth = params.MaxDepth
    net.Flush()


    model = Multicellularity_model(net=net, depth=depth, height= height, width= width, 
    initial_cells=initial_cells,
    energy = energy,
    step_count = step_count,
    nb_gap_junctions = nb_gap_junctions,
    fitness=fitness_function,
    goal = goal)
   

    model.verbose = False
    fitness_individual=0
    fitness_test = model.run_model(fitness_evaluation=True)
    if fitness_test>95:
        for i in range(10):
            fitness_individual += fitness_test
            fitness_test = model.run_model(fitness_evaluation=True)
            #net.Flush()
        fitness_individual = fitness_individual/10

        net.Save("./Results/" + file + "/winner_net_"+ str(fitness_individual)+".txt")
    else: 
        fitness_individual = fitness_test
        
    return fitness_individual
 



# If run as script.
if __name__ == '__main__':

# Result file
    myDatetime = datetime.datetime.now()
    myString = myDatetime.strftime('%Y-%m-%d %H:%M')
    file = myString.replace(' ','_')
    
    os.makedirs("./Results/" + file, exist_ok=True)
    
    # Save general params
    Result_file = "./Results/" + file + "/"+ "general_params.txt"
    
    with open("general_params.txt","w+") as general_params_file:
        print("nb_gens: %s \n\
depth: %s  \n\
height: %s  \n\
width: %s  \n\
max_fitness: %s  \n\
energy: %s \n\
nb_gap_junctions: %s  \n\
step_count: %s" % (nb_gens, depth, height, width, 
        max_fitness, energy, nb_gap_junctions, step_count), 
        file=general_params_file)
        
    general_params_file.close() 
    os.replace("general_params.txt", Result_file)
    shutil.copyfile("analysis.py",  "./Results/" + file + "/" + "analysis.py")
    shutil.copyfile("run.py",  "./Results/" + file + "/" + "run.py")
    shutil.copytree('./multicellularity', "./Results/" + file + "/" + "multicellularity")
    

    # random seed
    seed = int(time.time()) #1660341957#
    np.save("./Results/" + file + "/seed", seed)
    params.Save("./Results/" + file + "/multiNEAT_params.txt")
    
    genome = NEAT.Genome(0,
                    substrate.GetMinCPPNInputs(),
                    2, # hidden units
                    substrate.GetMinCPPNOutputs(),
                    False,
                    NEAT.ActivationFunction.TANH,
                    NEAT.ActivationFunction.SIGNED_GAUSS,
                    1, # hidden layers seed
                    params, 
                    1)  # one hidden layer
    
    pop = NEAT.Population(genome, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    # Run for up to N generations.
    start_time = time.time()
    best_genome_ser = None
    best_ever_goal_fitness = -20000
    best_id = -1
    solution_found = False
    plot_best_fitness=[]
    
    for generation in range(nb_gens):
        
        gen_time = time.time()

        # Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)

        fitnesses = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        
        # Store the best genome
        solution_found = max(fitnesses) >= max_fitness
        gen_best_genome = pop.Species[0].GetLeader()   
        gen_best_ID = gen_best_genome.GetID()
        if solution_found or best_ever_goal_fitness < max(fitnesses):
            best_genome_ser = pickle.dumps(gen_best_genome) # dump to pickle to freeze the genome state
            best_ever_goal_fitness = max(fitnesses)
            best_id = gen_best_genome.GetID()

        if solution_found:
            print('Solution found at generation: %d, best fitness: %f, species count: %d' % (generation, max(fitnesses), len(pop.Species)))
            break
        
        # advance to the next generation
        pop.Epoch()
        plot_best_fitness.append(max(fitnesses))

        # print statistics        
        gen_elapsed_time = time.time() - gen_time
        print("")
        print('*******************************************')
        print("Generation: %d" % generation)
        print("Best fitness: %f, genome ID: %d" % (max(fitnesses), gen_best_ID))
        print("Species count: %d" % len(pop.Species))
        print("Generation elapsed time: %.3f sec" % (gen_elapsed_time))
        print("Best fitness ever: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
        print('*******************************************')
        print("")


    elapsed_time = time.time() - start_time
    best_genome = pickle.loads(best_genome_ser)
    # write best genome to the file
    best_genome_file = os.path.join(".", "best_genome.pickle")
    np.save('plot_best_fitness', plot_best_fitness)
    
    with open(best_genome_file, 'wb') as genome_file:
        pickle.dump(best_genome, genome_file, pickle.HIGHEST_PROTOCOL)
    genome_file.close()
    os.replace(best_genome_file, "./Results/" + file + "/" + "best_genome.pickle")
        



    # Print experiment statistics
    print("\nBest ever fitness: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
    print("\nTrial elapsed time: %.3f sec" % (elapsed_time))
    print("Random seed:", seed)
    


        
    # Visualize best network's Genome
    winner_net_CPPN = NEAT.NeuralNetwork()
    gen_best_genome.BuildPhenotype(winner_net_CPPN)
    #pop.Species[0].GetLeader().BuildPhenotype(winner_net_CPPN)
    # img = np.zeros((500, 500, 3), dtype=np.uint8)
    # img += 10
    # NEAT.DrawPhenotype(img, (0, 0, 500, 500), winner_net_CPPN)
    # cv2.imshow("CPPN", img)
    # cv2.waitKey(0)
    #visualize.draw_net(winner_net_CPPN, view=False, node_names=None, filename="winner_net_CPPN", directory="./Results/" + file + "/", fmt='pdf')
    print("\nCPPN nodes: %d, connections: %d" % (len(winner_net_CPPN.neurons), len(winner_net_CPPN.connections)))

    # Visualize best network's phenotype
    winner_net = NEAT.NeuralNetwork()
    gen_best_genome.BuildESHyperNEATPhenotype(winner_net, substrate, params)
    winner_net.Save("./Results/" + file + "/winner_net.txt")
    


    with open("score.txt","w+") as score_file:
        score_file.write("\nBest ever fitness: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
        score_file.write("\nTrial elapsed time: %.3f sec" % (elapsed_time))
        score_file.write("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))
    score_file.close() 
    os.replace("score.txt", "./Results/" + file + "/" + "score.txt")



    #visualize.draw_net(winner_net, view=False, node_names=None, filename="substrate_graph", directory="./Results/" + file + "/", fmt='pdf')
    print("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))
    
    #save_params
    params_file = os.path.join(".", "params.pickle")
    with open(params_file, 'wb') as param_file1:
        pickle.dump(params, param_file1)
    param_file1.close() 
    os.replace(params_file, "./Results/" + file + "/" +  "params.pickle")

    
    #save_substrate 
    substrate_file = os.path.join(".", "substrate.pickle")
    with open(substrate_file, 'wb') as substrate_file1:
        pickle.dump(substrate, substrate_file1)
    substrate_file1.close() 
    os.replace(substrate_file, "./Results/" + file + "/" + "substrate.pickle")

    print('*******************************************')
    print("EVOLUTION DONE")
    print('*******************************************')
    
    
    import subprocess
    subprocess.Popen("python3 analysis.py", cwd="Results/" + file + "/", shell=True)
    os.rename('Results/' + file,'Results/' + file +'_'+ str(int(best_ever_goal_fitness)))
















