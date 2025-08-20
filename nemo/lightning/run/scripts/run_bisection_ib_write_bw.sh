#!/bin/bash


# printenv | sort   # In case we need to dump all env variables

export INITIAL_PORT_NUMBER=13000
export PREFIX="${SLURM_NODEID}-${SLURM_LOCALID}:"
AMOUNT_OF_QPS=4
PACKET_SIZE=65536

# TODO: get the ib devices from the topology
# mlx5_2 and mlx5_11 are allocates for north/south traffic
IB_DEVICES=("mlx5_0" "mlx5_1" "mlx5_3" "mlx5_4" "mlx5_5" "mlx5_6" "mlx5_7" "mlx5_8" "mlx5_9" "mlx5_10" "mlx5_12" "mlx5_13" "mlx5_14" "mlx5_15" "mlx5_16" "mlx5_17")

IFS=',' read -ra NODES <<< "$NODES_STR"
echo "${PREFIX} NODES: ${NODES[@]}"


# echo "======================================================================"
# echo "All nodes: ${NODES[@]}"
# echo "Queue pairs per connection: $AMOUNT_OF_QPS"
# echo "Packet size: $PACKET_SIZE bytes"
# echo "Job ID: $SLURM_JOB_ID"
# echo "Nodes allocated: $SLURM_STEP_NUM_NODES"
# echo "Start time: $(date)"
# echo "======================================================================"

# extract the params from the local rank:
# the lsb spefifies if it's server or client
# the rest of the bits specify the device index
localrank=$((SLURM_LOCALID))
is_server=$((localrank & 1))
device_index=$((localrank >> 1))


# get the ib device from the device index
ib_device=${IB_DEVICES[$device_index]}

PORT=$((INITIAL_PORT_NUMBER + device_index))
echo "${PREFIX} Local rank: $localrank, is_server: $is_server, device_index: $device_index, IB device: $ib_device, Port: $PORT"


# find the number of nodes divided by 2
PAIR_NODE_OFFSET=$((SLURM_STEP_NUM_NODES / 2))
PAIR_NODE_INDEX=$(( (SLURM_NODEID + PAIR_NODE_OFFSET) % SLURM_STEP_NUM_NODES ))
PAIR_NODE=${NODES[$PAIR_NODE_INDEX]}


echo "${PREFIX} SLURM_STEP_NUM_NODES: $SLURM_STEP_NUM_NODES, PAIR_NODE_OFFSET: $PAIR_NODE_OFFSET, Pair node index: $PAIR_NODE_INDEX, node: $PAIR_NODE"



if [ $is_server -eq 1 ]; then
  echo "${PREFIX} Starting server on $SLURMD_NODENAME:$ib_device"
  while ! ib_write_bw -d $ib_device -s $PACKET_SIZE --report_gbits --qp=$AMOUNT_OF_QPS -p $PORT --run_infinitely -F -x 3 -v --use_cuda=0 --tclass=41; do
    echo "${PREFIX} Server failed to start, retrying..."
    sleep 5
  done
else
  echo "${PREFIX} waiting for server on $PAIR_NODE:$ib_device to start"
  sleep 30
  echo "${PREFIX} Starting client sending from $SLURMD_NODENAME:$ib_device to $PAIR_NODE:$ib_device"
  while ! ib_write_bw -d $ib_device -s $PACKET_SIZE --report_gbits --qp=$AMOUNT_OF_QPS -p $PORT --run_infinitely -F -x 3 -v ${PAIR_NODE} --use_cuda=0 --tclass=41; do
    echo "${PREFIX} Client failed to start, retrying..."
    sleep 5
  done
fi
