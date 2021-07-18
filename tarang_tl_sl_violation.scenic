import sys
try:
#    sys.path.append("./libs/carla-0.9.9-py3.7-linux-x86_64.egg")
    sys.path.append("./libs/carla-0.9.11-py3.7-linux-x86_64.egg")

except IndexError:
    pass
from random import random

""" Scenario Description
Background Activity
The simulation is filled with vehicles that freely roam around the town. 
The Traffic Manager is configured for vehicles to abnormal behavior via traffic lights and speed limits violations
"""

param map = localPath('maps/CARLA/Town03.xodr')    # or other CARLA map that definitely works
param carla_map = 'Town03'
param weather = 'ClearNoon' 
model scenic.simulators.carla.model

VEHICLE_PERC_SPEED_DIFF = -30
SPEED_VIOLATION_PROB = 60 # probability that a vehicle is a speed limit violator(always exceeds speed limit)
TL_VIOLATION_PROB = 70  # probability with which the vehicle may violate a traffic light - this prob is constant for all vehicles

# Background activity
background_vehicles = []
for _ in range(25):
    lane = Uniform(*network.lanes)
    spot = OrientedPoint on lane.centerline
    kwargs = {"ignore_lights_percentage":TL_VIOLATION_PROB}
    if (random()*100) < SPEED_VIOLATION_PROB:
        kwargs["vehicle_percentage_speed_difference"] = VEHICLE_PERC_SPEED_DIFF
    background_car = Car at spot,
        with behavior AutopilotBehavior(**kwargs)
    background_vehicles.append(background_car)

kwargs = {"ignore_lights_percentage":TL_VIOLATION_PROB}
if (random()*100) < SPEED_VIOLATION_PROB:
    kwargs["vehicle_percentage_speed_difference"] = VEHICLE_PERC_SPEED_DIFF
ego = Car following roadDirection from spot for Range(-30, -20),
    with behavior AutopilotBehavior(**kwargs)