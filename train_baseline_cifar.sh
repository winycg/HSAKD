python train_baseline.py --arch ShuffleV1 --data ./data/ --init-lr 0.01 --gpu 0
python train_baseline.py --arch ShuffleV2 --data ./data/ --init-lr 0.01 --gpu 0
python train_baseline.py --arch mobilenetV2 --data ./data/ --init-lr 0.01 --gpu 0


python train_baseline.py --arch wrn_16_2 --data ./data/  --gpu 0
python train_baseline.py --arch wrn_40_1 --data ./data/  --gpu 0
python train_baseline.py --arch resnet8x4 --data ./data/  --gpu 0
python train_baseline.py --arch resnet20 --data ./data/  --gpu 0


python train_baseline_cifar.py --arch mobilenetV2 --data /home/ycg/hhd/dataset/ --init-lr 0.01 --gpu 0
python train_baseline_cifar.py --arch wrn_16_2 --data /home/ycg/hhd/dataset/  --gpu 0