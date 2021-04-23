

SCENIC_FILE=$1
NUM_SCENARIOS=$2
RECORDING_FILE=`realpath $3`
MAX_TIME=$4
SCENIC_SEED=$5

scenic $SCENIC_FILE \
    --simulate -b\
    --model scenic.simulators.carla.model \
    --time $MAX_TIME \
    --count $NUM_SCENARIOS \
    --seed $SCENIC_SEED \
    --param record $RECORDING_FILE