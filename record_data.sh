CARLA_PATH="/home/tarang/Code/carla994" # path to the folder where you exracted carla - Update This
NUM_SCENARIOS=20 # number of scenarios to record
SCENARIO_MAX_TIME=100

# Initialize carla
$CARLA_PATH/CarlaUE4.sh -opengl &
PID=$!
echo "Carla PID=$PID"
sleep 4 # sleep to ensure Carla has opened up and is initialized

# Run Scenic Scenarios

echo "Running Scenic for $NUM_SCENARIOS Debris avoidance scenarios"
./run_scenic.sh tarang_test_debris.scenic $NUM_SCENARIOS debris_avoidance_recordings $SCENARIO_MAX_TIME
echo "Running Scenic for $NUM_SCENARIOS Debris avoidance scenarios"
./run_scenic.sh tarang_test_opposing_car.scenic $NUM_SCENARIOS oncoming_car_recordings $SCENARIO_MAX_TIME
echo "Running Scenic for $NUM_SCENARIOS Normal scenarios"
./run_scenic.sh tarang_test_normal.scenic $NUM_SCENARIOS normal_recordings 500
echo "Done Scenic task"

python recorder.py
echo "Done Python task"

pkill -f "CarlaUE4" 
