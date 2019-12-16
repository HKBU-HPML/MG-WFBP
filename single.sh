#!/bin/bash
dnn="${dnn:-resnet50}"
source exp_configs/$dnn.conf
nstepsupdate=1
PY=python
$PY dl_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate
