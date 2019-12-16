dnns=( "resnet50" "googlenet" "resnet152" "inceptionv4" "densenet161" "densenet201" )
for dnn in "${dnns[@]}"
do
    dnn=$dnn max_epochs=1 ./single.sh
done
