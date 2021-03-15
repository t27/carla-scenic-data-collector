


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
    - `generate_segment_trajectories` - Convert the individual agent level data to other model specific formats(mainly used for the dtw maps right now)
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


## TODO: Add the modelling notes and howto