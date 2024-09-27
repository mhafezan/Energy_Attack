

CIFAR_DATA=/home/atoofian/projects/def-atoofian/atoofian/TrojAI/cifar100/old_GoogleNet_train_test/train



for i in {1..200}
do
    echo "********************************$i**********************************"

    python3 ./test.py   -net googlenet -gpu -b 10      -dataset $CIFAR_DATA/data  -weights   ./weight_dir/googlenet_cifar100_$i.pkl  

    echo "********************************************************************"

done


