dnns=( "resnet50" "googlenet" "resnet152" "inceptionv4" "densenet161" "densenet201" )
thresholds=( "524288000" "0" ) 
ns=( "16" "8" "4" "2" )
max_epochs=1
for dnn in "${dnns[@]}"
do
    for thres in "${thresholds[@]}"
    do
        for nworkers in "${ns[@]}"
        do
            for compressor in "${compressors[@]}"
            do
                lr=0.8 max_epochs=$max_epochs dnn=$dnn threshold=$thres nworkers=$nworkers ./horovod_mpi_nvcluster.sh
            done
        done
    done
done
