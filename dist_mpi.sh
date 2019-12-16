#!/bin/bash
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
threshold="${threshold:-0}"
nwpernode=4
nstepsupdate=1
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=python
GRADSPATH=./logs

$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_IB_DISABLE=1 \
    $PY dist_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --threshold $threshold --saved-dir $GRADSPATH 


# 10GbE configurations
#    -mca pml ob1 -mca btl ^openib \
#    -mca btl_tcp_if_include eth0 \
#    -x NCCL_DEBUG=INFO  \
#    -x NCCL_SOCKET_IFNAME=eth0 \
#    -x NCCL_IB_DISABLE=1 \

# 56GbIB configurations
#    -mca pml ob1 -mca btl openib \
#    -mca btl_openib_allow_ib 1\
#    -x NCCL_DEBUG=INFO  \
#    -x NCCL_IB_DISABLE=0 \
