
import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
import torch



class cub200():
    def __init__(self,root,is_train=True,data_len=None,transform=None,target_transform=None):
        self.root=root
        self.is_train=is_train
        self.transform=transform
        self.target_transform=target_transform
        img_txt_file=open(os.path.join(self.root,'images.txt'))
        label_txt_file=open(os.path.join(self.root,'image_class_labels.txt'))
        train_val_file=open(os.path.join(self.root,'train_test_split.txt'))#先写好路径 Specify the path first.

        img_name_list=[]
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])#去掉换行符 删除了前面的序列号然后只取最后的单词，也就是图片路径，Remove newline characters, strip out the preceding sequence number, and extract only the last word, which is the image path.
        label_list=[]
        for line in label_txt_file:
            label_list.append(line[:-1].split(' ')[-1])#和上面一样，注意line[:-1]的意思是一直取直到倒数第二个字符，也就是为了删除换行符

        train_test_list=[]
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))#加载训练测试集的路径 Load paths for training and testing sets.

        train_file_list=[x for i,x in zip(train_test_list,img_name_list)if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]#将图片和布尔值匹配，来决定哪些是测试集哪些是训练集 Match images with boolean values to determine which ones are for the test set and which ones are for the training set.

        train_label_list = [x for i, x in zip(train_test_list, label_list) if i][:data_len]#同理，保留的是label Similarly, preserve the labels.
        test_label_list = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        if self.is_train:
            self.train_img=[scipy.misc.imread(os.path.join(self.root,'images',train_file))for train_file in train_file_list[:data_len]]#将 train_file_list 中的前 data_len 个文件读入，并将其图像数据存储在 self.train_img 列表中
            self.train_label=train_label_list
            self.val_img=[scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in test_file_list[:data_len]]#同理将测试集写入
            self.val_label=test_label_list

        def __getitem__(self,index):
            if self.is_train:
                img,target=self.train_img[index],self.train_label[index]
            else:
                img,target=self.test_img[index],self.test_label[index]

            if len(img.shape)==2:
                img=np.stack([img]*3,2)

            img=Image.fromarray(img,mode='RGB')
            if self.transform is not None:
                target=self.target_transform(target)
            return img,target

        def __len__(self):
            if self.is_train:
                return len(self.train_lable)
            else:
                return len(self.test_label)

class CubTrain(cub200):
    def __init__(self,size,training_images_list_file):
        super().__init__()
        with open(training_images_list_file):
