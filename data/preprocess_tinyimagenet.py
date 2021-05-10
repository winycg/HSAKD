import os

os.system('mv ./tiny-imagenet-200/val ./tiny-imagenet-200/val_original')
val_labels_t = []
val_labels = []
val_names = []
with open('./tiny-imagenet-200/val_original/val_annotations.txt') as txt:
    for line in txt:
        img_name = line.strip('\n').split('\t')[0]
        label_name = line.strip('\n').split('\t')[1]

        if not os.path.isdir('./tiny-imagenet-200/val/'+label_name):
            os.makedirs('./tiny-imagenet-200/val/'+label_name)
        os.system('cp ./tiny-imagenet-200/val_original/images/'+img_name+' ./tiny-imagenet-200/val/'+label_name)