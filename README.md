


# Getting Started

Prerequisites(Ubuntu 20.04/18.04)

1. Install [Carla 0.9.9.4](https://github.com/carla-simulator/carla/releases/tag/0.9.9) ([direct download](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.9.4.tar.gz)) (newer versions should work too, but the code is tested with 0.9.9.4)
    1. Use the tar.gz instead of the `apt` method to ensure we can access the carla/PythonAPI folder without changing too many permissions
2. Python environment
    1. Python 3.8
    2. Set up the the Carla Python API - ensure the Carla Python API egg file is present in `libs/`. If you install a Carla version that is not 0.9.9, ensure to copy the `python3.x` egg file from the `PythonAPI/carla/dist` folder in your Carla directory to the `libs` folder here. Also ensure you update the path to the egg file in the `recorder.py` script
    3. Use the requirements file to install other dependencies
        1. `pip install -r requirements.txt`
    4. For the modelling section, use the requirements file in the `modelling` folder

# Description of the files

1. Scenic scenes
    - `debris.scenic` - Builds a scenic scenario for a debris avoidance scenario. Also adds one spectator vehicle.
    - `opposing_car.scenic` - - Builds a scenic scenario for a scenario where the ego is facing an oncoming car(which is violating lane rules). Also adds one spectator vehicle.
    - `platoon.scenic` - A basic platoon/convoy scenario. This file just sets up the scene, no specific behavior has been added yet.

2. Runners - bash files
    - `record_data.sh` - This is the overall script that runs the scenario in Carla, records the data and ensures the binary carla recordings are converted to a more easy to use csv. The binary carla to csv conversion requires the use of Carla hence both the parts need the Carla simulator.
    - `run_scenic.sh` - This runs the `.scenic` scenarios and stores the data. Only use this if you need to test anything specific to scenic

3. Python Scripts
    - `generate_agent_maps` - Convert the scenario CSV to individual agent and frame level data - This is usually time consuming, but it helps in the modelling related dataprep later
    - `generate_segment_trajectories` - Convert the individual agent level data to other model specific formats(used for the dtw maps and the random forest based classifier as well)
    - `traffic_light_and_speed_limit_recorder` - a Carla Python API based traffic light violation and speed limit violation data generator. The current feature set of scenic does not all for the specific traffic manager control that we need hence this is used now. The plan is to add those features into scenic so that we can have a `.scenic` file with the traffic light and speed limit violating agents
    - `visualize_agent_maps` - visualizes the maps created in 'generate_agent_maps'

4. `modelling` and `playground`
    - We have 2 modelling approaches a basic random forest and an autoencoder
    - Both the processes involve some model specific data preparation and evaluation
    - TODO: Add descriptions of the model code and the jupyter notebooks 

# How to run and collect data

1. Setup Carla path and number of scenarios to record
    - Add the carla path and the number of scenarios to run to the top of `record_data.sh`
2. From the current project folder, run `record_data.sh` 
    - The data will be stored in the relevant recording folders
3. We can now proceed to use this scenario level data for the modelling tasks


# About the anomalous scenarios

1. Debris Avoidance
2. Oncoming Car
3. Traffic Light Violation
4. Speed Limit Violation
# Modelling Approaches

We include 2 ways to use the data collected in models. The modelling code also includes the data preparation steps. Also, the data preparation code(`generate_agent_maps.py` and `generate_segment_trajectories.py`) is used in various steps of the modelling. Both the following modelling tasks have some prerequisites on the data, which are described in detail in the modelling subfolder. [Modelling Readme](./modelling/)

## Random Forest Based Frame Classifier

This classifier used frame level features extracted from each frame that we record. These frame level features are extracted from both _normal_ and _anomalous_ scenarios. 
Each frame results in one training record and is either _normal_ or _anomalous_. We then collect all the frames, shuffle them and formulate the problem as a supervised classification task.
We use a random forest and a MLP Classifier. We use the standard implementations of these models in the Scikit-learn library with relevant hyperparameters.

### Data Prep

The data prep for the frame classifier model is available in the `generate_segment_trajectories.py` file in the `get_basic_dataframe` function. This function requires the individual agent maps to be generated through the `generate_agent_maps.py` module. 

`get_basic_dataframe` parses through the agent maps and aggregates the frame level stats using the various helper functions.  

### Modelling

The modelling code is available in the [`modelling/frame_classifier/model.py`](modelling/frame_classifier/model.py) file. This file includes the modelling and visualization code. Also, more visualization code and examples can be found in the [`playground/basic_model.ipynb`](playground/basic_model.ipynb) notebook.


## DTW Auto Encoder

The DTW Autoencoder uses agent level dtw maps. The dtw maps consider the ego and each of the surrounding agents to generate a 2D dtw cost map. For each neighbor of the ego, we get a cost map. We then stack these maps to generate a 3D stack of maps. The maximum height of the stack is fixed. If we have less number of neighbors than the maximum, we pad the stack with zeros. 

This DTW map is used in a Convolutional Autoencoder. First, we train the autoencoder with all the _normal_ scenario data. Now, once the autoencoder is trained, we can use the encoder to encode any scenario into a single vector.

### Data Prep

For the autoencoder the main process involves building the DTW map tensor. This is done in the `generate_segment_trajectories.py` file in the `get_dtw_map` function. 

### Modelling

The model, the dataloader, the training code and the inference code can be found in the `modelling/dtw_autoencoder` folder. Also, in the inference, we cache the vectors generated from the encoder and use them for further modelling or visualization. These can be found in the playground as well - [`playground/dtw_modelling_data_experiments.ipynb`](playground/dtw_modelling_data_experiments.ipynb), [`playground/embedding_classifier.ipynb`](playground/embedding_classifier.ipynb) and [`playground/embedding_vis.ipynb`](playground/embedding_vis.ipynb).

## Agent Maps

Also, we can visualize the agent maps separately as well. This visualization is a simple 2D visualization of the agent maps. The `visualize_agent_maps.py` includes all the code to visualize these agent maps.
By default the code will popup a window showing the visualization and also store a video in the current directory.
