# capture data for 100 rounds for each scenario file

# CARLA_PATH="/home/tarang/Code/carla994" # path to the folder where you exracted carla - Update This
CARLA_PATH="/home/tarang/Code/CARLA_0.9.11" # path to the folder where you exracted carla - Update This

NUM_SCENARIOS=30 # number of scenarios to record
NUM_REAL_SCENARIOS=50
SCENARIO_MAX_TIME=100 # in steps. 100 steps ~=7-8s
SCENIC_SEED=72

# Initialize carla
$CARLA_PATH/CarlaUE4.sh -opengl &
PID=$!
echo "Carla PID=$PID"
sleep 4 # sleep to ensure Carla has opened up and is initialized

# Run Scenic Scenarios

echo "Running Scenic for $NUM_SCENARIOS Debris avoidance scenarios"
./scripts/run_scenic.sh debris.scenic $NUM_SCENARIOS test_debris_avoidance_recordings $SCENARIO_MAX_TIME $SCENIC_SEED
echo "Running Scenic for $NUM_SCENARIOS Oncoming Car scenarios"
./scripts/run_scenic.sh oncoming_car.scenic $NUM_SCENARIOS test_oncoming_car_recordings $SCENARIO_MAX_TIME $SCENIC_SEED

# the below 2 scenic files are not currently used, we used the python files in record_tl_sl.sh and record_nominal.sh instead
# echo "Running Scenic for $NUM_SCENARIOS Traffic Light and Speed Limit violation scenarios"
# ./run_scenic.sh tarang_tl_sl_violation.scenic $NUM_SCENARIOS tl_sl_recordings $SCENARIO_MAX_TIME $SCENIC_SEED
# echo "Running Scenic for $NUM_REAL_SCENARIOS Nominal scenarios"
# ./run_scenic.sh tarang_test_normal.scenic $NUM_REAL_SCENARIOS normal_recordings 500 $SCENIC_SEED
echo "Done Scenic task"

# convert Scenic's output (Carla logs to CSV/Parquet files)
python recorder.py --test 

# Record Traffic Light and SpeedLimit violations
echo "Running CARLA API task for $NUM_SCENARIOS TL and SL violation scenarios"
python carla_python_api_recorder.py -s tl_sl -n $NUM_SCENARIOS --test

echo "Running CARLA API task for $NUM_REAL_SCENARIOS nominal scenarios"
python carla_python_api_recorder.py -s nominal -n $NUM_REAL_SCENARIOS --test


echo "Done Python task"

pkill -f "CarlaUE4" 
