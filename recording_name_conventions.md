# Recording naming conventions


- Each base recording name will have a "Round name". The round name is essentially the filename. 
- We use the round name during the data conversion process as well
- Step 1 (generate_agent_maps.py) uses a list of the round names and folders to convert the raw data to agent_maps. The agent maps are stored in a folder called `agent_maps`. The original parent folder of the raw file is used inside the `agent_maps` folder as well. For example. when converting `anomaly_type1/scenario1.csv`  to agent maps, the resulting agent and frame level agent maps will be in `agent_maps/anomaly_type1/*.csv.gz`, where `*` is the filename.
    - the filename of the agent map includes the round name, the agent id and the frame id
    - csv.gz was mainly chosen for the storage constraint. Since we have a lot of agent maps, the size gets much bigger. Both csv and parquet were larger than a csv.gz file
- Each of the modelling data prep functions have an `agent_map_folder` path. This should be the full path of the subfolder. For example, when you need to convert `anomaly_type1`, the `agent_map_folder` argument should be `agent_maps/anomaly_type1/`
