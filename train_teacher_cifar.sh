python train_teacher_cifar.py \
    --arch wrn_40_2_aux \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 2 --manual 0

python train_teacher_cifar.py \
    --arch ResNet50_aux \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 2 --manual 0

python train_teacher_cifar.py \
    --arch vgg13_bn_aux \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 2 --manual 0

python train_teacher_cifar.py \
    --arch resnet56_aux \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 2 --manual 0

python train_teacher_cifar.py \
    --arch resnet32x4_aux \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 2 --manual 0