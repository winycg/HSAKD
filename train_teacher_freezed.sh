python train_teacher_cifar.py \
    --arch wrn_40_2_aux \
    --milestones 30 60 90 --epochs 100 \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 2 --manual 0 \
    --pretrained-backbone ./pretrained_backbones/wrn_40_2.pth \
    --freezed

python train_teacher_cifar.py \
    --arch resnet56_aux \
    --milestones 30 60 90 --epochs 100 \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 2 --manual 0 \
    --pretrained-backbone ./pretrained_backbones/resnet56.pth \
    --freezed

python train_teacher_cifar.py \
    --arch ResNet50_aux \
    --milestones 30 60 90 --epochs 100 \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 2 --manual 0 \
    --pretrained-backbone ./pretrained_backbones/ResNet50.pth \
    --freezed

python train_teacher_cifar.py \
    --arch vgg13_bn_aux \
    --milestones 30 60 90 --epochs 100 \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 2 --manual 0 \
    --pretrained-backbone ./pretrained_backbones/vgg13_bn.pth \
    --freezed

python train_teacher_cifar.py \
    --arch resnet32x4_aux \
    --milestones 30 60 90 --epochs 100 \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 2 --manual 0 \
    --pretrained-backbone ./pretrained_backbones/resnet32x4.pth \
    --freezed