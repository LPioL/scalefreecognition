from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from multicellularity.agents import Cell, Toxine, Food
from multicellularity.model import Multicellularity_model


def agents_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.state ==0:
        portrayal["Color"] = ["grey"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.state ==1:
        portrayal["Color"] = ["blue"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.state ==2:
        portrayal["Color"] = ["red"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["r"] = 1
        
    elif type(agent) is Toxine:
        portrayal["Color"] = ["red"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["w"] = 0.5
        portrayal["h"] = 0.5

    elif type(agent) is Food:
        portrayal["Color"] = ["#84e184", "#adebad", "#d6f5d6"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["w"] = 1
        portrayal["h"] = 1
    return portrayal


canvas_element = CanvasGrid(agents_portrayal, height, width, 800, 800)
chart_element = ChartModule(
    [{"Label": "Cells", "Color": "#AA0000"}], [{"Label": "Food", "Color": "##84e184"}],
    [{"Label": "Cells", "Color": "#AA0000"}], [{"Label": "Food", "Color": "##84e184"}]

)

model_params = {
    "initial_cell": UserSettableParameter(
        "slider", "Initial Cell Population", 100, 10, 300
    ),
    "initial_food": UserSettableParameter(
        "slider", "Initial Food frequency", 50, 10, 300
    ),
    "initial_toxines": UserSettableParameter(
        "slider", "Initial Toxines frequency", 50, 10, 300
    ),
    "cell_gain_from_food": UserSettableParameter(
        "slider", "Cell Gain From Food Rate", 4, 1, 50
    )
}

server = ModularServer(
    Multicellularity_model, [canvas_element, chart_element], "Multi-cellularity", model_params
)
server.port = 8521
