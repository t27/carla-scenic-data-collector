

SCENIC_FILE=$1
NUM_SCENARIOS=$2
RECORDING_FILE=`realpath $3`

scenic $SCENIC_FILE \
    --simulate -b\
    --model scenic.simulators.carla.model \
    --time 100 \
    --count $NUM_SCENARIOS \
    --seed 27 \
    --param record $RECORDING_FILE