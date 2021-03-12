

## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath('maps/CARLA/Town03.xodr')  # or other CARLA map that definitely works

param carla_map = 'Town03'
param weather = 'ClearNoon' 
# param record = './debris_avoidance_recordings'
model scenic.simulators.carla.model #located in scenic/simulators/carla/model.scenic


## CONSTANTS
MAX_BREAK_THRESHOLD = 1
SAFETY_DISTANCE = 8
PARKING_SIDEWALK_OFFSET_RANGE = 2
CUT_IN_TRIGGER_DISTANCE = Range(10, 12)
EGO_SPEED = 8
PARKEDCAR_SPEED = 7

DIST_THRESHOLD = 15
BYPASS_DIST = 15
roads = network.roads
## DEFINING BEHAVIORS
behavior CutInBehavior(laneToFollow, target_speed):
    while (distance from self to ego) > CUT_IN_TRIGGER_DISTANCE:
        wait

    do FollowLaneBehavior(laneToFollow = laneToFollow, target_speed=target_speed)

"""
behavior CollisionAvoidance():
    while withinDistanceToAnyObjs(self, SAFETY_DISTANCE):
        take SetBrakeAction(MAX_BREAK_THRESHOLD)

behavior EgoBehavior(target_speed):
    try: 
        do FollowLaneBehavior(target_speed=target_speed)

    interrupt when withinDistanceToAnyObjs(self, SAFETY_DISTANCE):
        do CollisionAvoidance()
"""

## BEHAVIORS
behavior EgoBehavior(speed=10):
    try: 
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyObjs(self, DIST_THRESHOLD):

        # change to left (overtaking)
        faster_lane = self.laneSection.fasterLane
        print(faster_lane)
        do LaneChangeBehavior(laneSectionToSwitch=faster_lane, target_speed=speed)
        do FollowLaneBehavior(speed, laneToFollow=faster_lane.lane) for 5 seconds

        # change to right
        slower_lane = self.laneSection.slowerLane
        do LaneChangeBehavior(laneSectionToSwitch=slower_lane, target_speed=speed)
        do FollowLaneBehavior(speed) for 5 seconds
        terminate


# find lanes that have a valid left lane in same direction
# lanes_with_left_lane = filter(lambda s: s._laneToLeft is not None, network.laneSections)
# assert len(lanes_with_left_lane) > 0, \
#     'No lane sections with adjacent left lane in network.'
# intersec = Uniform(*network.intersections)
# ego_lane = Uniform(*intersec.outgoingLanes)
# make sure to put '*' to uniformly randomly select from all elements of the list of roads
select_road = Uniform(*roads)

# in roads.py, the 'class Road' contains 'lanes' which is a list of lanes whose rightmost lane is indexed 0
ego_lane = select_road.lanes[0]

ego = Car on ego_lane.centerline,
        with behavior EgoBehavior(speed=EGO_SPEED)


# can also use "Prop" here
debris1 = Trash following roadDirection for Range(16, 25)
#debris2 = Debris following roadDirection from debris1 for Range(5, 10)
#debris3 = Trash following roadDirection from debris2 for Range(5, 10)
require (distance to intersection) > 50
require (debris1.laneSection._fasterLane is not None)

number_of_agents = 5

# #TODO Spawn vehicles in nearby lanes instead of randomly
# background_vehicles = []
# valid = True
# for _ in range(number_of_agents):
#     lane = Uniform(*network.lanes)
#     if lane is ego_lane:
#         print("same lane")
#         continue
#     spot = OrientedPoint on lane.centerline
#     background_car = Car at spot,
#         with behavior AutopilotBehavior()
#     #assert (distance from background_car to ego) < 2000
#     #assert ego.lane != background_car.lane
#     background_vehicles.append(background_car)

# for bg_car in background_vehicles:
#     require bg_car can see ego

#TODO Spawn vehicles in nearby lanes instead of randomly
valid = True
lane1 = Uniform(*select_road.lanes[1:])
require lane1 is not ego_lane
spot1 = OrientedPoint on lane1.centerline
background_car1 = Car at spot1,
    with behavior AutopilotBehavior()
require background_car1 can see ego


# lane2 = Uniform(*select_road.lanes[1:])
# require lane2 is not ego_lane
# spot2 = OrientedPoint on lane2.centerline
# background_car2 = Car at spot2,
#     with behavior AutopilotBehavior()
# require background_car2 can see ego



