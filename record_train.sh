# capture data for 100 rounds for each scenario file

CARLA_PATH="/home/tarang/Code/carla994" # path to the folder where you exracted carla - Update This
NUM_SCENARIOS=100 # number of scenarios to record
NUM_REAL_SCENARIOS=50
NUM_TL_SL_SCENARIOS=20

SCENARIO_MAX_TIME=100
SCENIC_SEED=27
# Initialize carla
$CARLA_PATH/CarlaUE4.sh -opengl &
PID=$!
echo "Carla PID=$PID"
sleep 4 # sleep to ensure Carla has opened up and is initialized

# Run Scenic Scenarios

echo "Running Scenic for $NUM_SCENARIOS Debris avoidance scenarios"
./scripts/run_scenic.sh tarang_test_debris.scenic $NUM_SCENARIOS debris_avoidance_recordings $SCENARIO_MAX_TIME $SCENIC_SEED
echo "Running Scenic for $NUM_SCENARIOS Debris avoidance scenarios"
./scripts/run_scenic.sh tarang_test_opposing_car.scenic $NUM_SCENARIOS oncoming_car_recordings $SCENARIO_MAX_TIME $SCENIC_SEED

# the below 2 scenic files are not currently used, we use carla_python_api_recorder instead
# echo "Running Scenic for $NUM_SCENARIOS Traffic Light and Speed Limit violation scenarios"
# ./run_scenic.sh tarang_tl_sl_violation.scenic $NUM_SCENARIOS tl_sl_recordings $SCENARIO_MAX_TIME $SCENIC_SEED
# echo "Running Scenic for $NUM_REAL_SCENARIOS Nominal scenarios"
# ./run_scenic.sh tarang_test_normal.scenic $NUM_REAL_SCENARIOS normal_recordings 500 $SCENIC_SEED
echo "Done Scenic task"

# Record Traffic Light and SpeedLimit violations
echo "Running CARLA API task for $NUM_TL_SL_REAL_SCENARIOS TL and SL violation scenarios"
python carla_python_api_recorder.py -s tl_sl -n $NUM_TL_SL_REAL_SCENARIOS

echo "Running CARLA API task for $NUM_TL_SL_REAL_SCENARIOS nominal scenarios"
python carla_python_api_recorder.py -s nominal -n $NUM_TL_SL_REAL_SCENARIOS

python recorder.py
echo "Done Python task"

pkill -f "CarlaUE4" 
