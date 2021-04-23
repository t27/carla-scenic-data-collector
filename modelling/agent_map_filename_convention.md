# TODO 

Document the agent map filename conventions 

This is for an intermediate format hence shouldn't be very important for the average user.


All the maps are in the `agent_maps` folder. Within this folder we have subfolders for each scenario. Each subfolder represents one scenario type.

Within the sobfolders, we have multiple `csv.gz` files. Each CSV file contains data for one *frame* in the frame of reference of a given *vehicle*(this vehicle is the ego vehicle for this file). Each vehicle in the scenario is considered as an ego vehicle one by one. Hence if we have _F_ frames and _N_ vehicles, the total number of `csv.gz` files will be `F*N`

The `.csv.gz` files are named like,

```
<roundname>_vehicle_<vehicle_id>_frame_<frame_id>.csv.gz

```

Where,

- `roundname` - We capture multiple rounds for each type of scenario. This is the unique round name for the given scenario type.
- `vehicle_id` - This is the id for the current ego vehicle. We also include all other vehicles in the whole round but their positions are offsetted based on the current ego vehicle
- `frame_id` - This is the frame number for which all the data in this file is valid for

We also include a `roundnames.txt` file in the root of the scenario subfolder that includes all the rounds that have been captured. This file is written when the captured scenario data is converted to the agent maps. The main assumption here is that all the capture is done at once (all csv.gz files for all the rounds are captured together)


These files and filenames are used when we process the agent map data for the modelling baselines. If you aren't using any of the baselines and are using your own system to generate the data the agent_map generation code might not be directly useful.