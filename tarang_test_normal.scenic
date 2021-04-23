import sys
try:
    sys.path.append("./libs/carla-0.9.9-py3.7-linux-x86_64.egg")
except IndexError:
    pass


## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath('maps/CARLA/Town03.xodr')  
param carla_map = 'Town03'
param weather = 'ClearNoon' 
model scenic.simulators.carla.model #located in scenic/simulators/carla/model.scenic

## CONSTANTS
MAX_BREAK_THRESHOLD = 1
SAFETY_DISTANCE = 15
LIGHT_DIST = 5
BYPASS_DIST = 18
CAR_SPEED = 7
roads = network.roads
## DEFINING BEHAVIORS
behavior CollisionAvoidance():
    while withinDistanceToAnyObjs(self, SAFETY_DISTANCE):
        take SetBrakeAction(MAX_BREAK_THRESHOLD)

behavior NormalCarBehavior():
    try:
        do FollowLaneBehavior(CAR_SPEED)
    interrupt when withinDistanceToRedYellowTrafficLight(self,LIGHT_DIST):
        while withinDistanceToRedYellowTrafficLight(self, LIGHT_DIST):
            take SetBrakeAction(MAX_BREAK_THRESHOLD)
 #   interrupt when withinDistanceToObjsInLane(self, SAFETY_DISTANCE):
 #       while withinDistanceToObjsInLane(self, SAFETY_DISTANCE):
 #           take SetBrakeAction(MAX_BREAK_THRESHOLD)
    

# make sure to put '*' to uniformly randomly select from all elements of the list of roads
select_road = Uniform(*roads)
ego_lane = select_road.lanes[0]

start = OrientedPoint on ego_lane.centerline
ego = Car at start,
    with behavior NormalCarBehavior()


for i in range(10):
    select_road2 = Uniform(*roads)
    ego_lane2 = select_road2.lanes[0]

    start = OrientedPoint on ego_lane2.centerline
    Car at start,
        with behavior NormalCarBehavior()

