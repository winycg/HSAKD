python train_online_kd_cifar.py \
    --arch WRN_40_2_WRN_40_2 \
    --checkpoint-dir ./checkpoint 

python train_online_kd_cifar.py \
    --arch WRN_40_1_WRN_40_1 \
    --checkpoint-dir ./checkpoint 

python train_online_kd_cifar.py \
    --arch resnet56_resnet56 \
    --checkpoint-dir ./checkpoint 


python train_online_kd_cifar.py \
    --arch resnet32x4_resnet32x4 \
    --checkpoint-dir ./checkpoint 

python train_online_kd_cifar.py \
    --arch vgg13_vgg13 \
    --checkpoint-dir ./checkpoint 


python train_online_kd_cifar.py \
    --arch mbv2_mbv2 \
    --init-lr 0.01 \
    --checkpoint-dir ./checkpoint 


python train_online_kd_cifar.py \
    --arch shufflev1_shufflev1 \
    --init-lr 0.01 \
    --checkpoint-dir ./checkpoint 

python train_online_kd_cifar.py \
    --arch shufflev2_shufflev2 \
    --init-lr 0.01 \
    --checkpoint-dir ./checkpoint 
