from torch.utils.data import Dataset
from utils.cfg import config as cfg
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
import os
import glob

class HDataset(Dataset):
    def __init__(self, task_name, mode):
        self.data_dir = "./data/echocardiography"
        self.mode = mode
        self.task_name = task_name
        self.img, self.label = self.data_load()

        if self.mode == "train" :
            self.data_len = cfg.train_dsize
        elif self.mode == "validation" : 
            self.data_len = cfg.val_dsize
        else :
            self.data_len = len(self.img)
        print('{:s} {:s} Dataset Initialize Completed'.format(self.task_name, self.mode))            

    def data_load(self, ):
        img = list()
        label = list()
        self.max_h = 0
        self.max_w = 0
        img_list = sorted(glob.glob(os.path.join(self.data_dir, self.mode, self.task_name, '*.png')))
        label_list = sorted(glob.glob(os.path.join(self.data_dir, self.mode, self.task_name, '*.npy')))
        if self.mode == 'train' :
            img_list.extend(sorted(glob.glob(os.path.join(self.data_dir, "validation", self.task_name, '*.png'))))
            label_list.extend(sorted(glob.glob(os.path.join(self.data_dir, "validation", self.task_name, '*.npy'))))

        tf = transforms.ToTensor()
        i = 0

        for im_name, lab_name in zip(img_list, label_list) :
            raw_img = tf(Image.open(im_name))
            raw_label = tf(np.load(lab_name))
        
            height, width  = raw_label.shape[-2:]
            
            height = int(height/10)
            width = int(width/10)
        
            raw_img = raw_img[:, height:-height, width:-width]
            raw_label = raw_label[:, height:-height, width:-width]
        
            img += [raw_img]
            label += [raw_label]

            # image resize
            
            # left 10% / right 10% / top 10% / bottom 10%
            
            h, w = label[-1].shape[-2:]

            if h > self.max_h : self.max_h = h
            if w > self.max_w : self.max_w = w
            i += 1

        print("Total {:d} data loaded".format(i))


        return img, label

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

    def __len__(self):
        return self.data_len

    def collate_fn(self, batch):
        f_data = []
        f_label = []
        tr = transforms.Resize((self.max_h, self.max_w))

        for data, label in batch:
            shape = data.shape
            # p2_r, p2_l, p2_u, p2_d = 0, 0, 0, 0
            # if shape[1] < self.max_h :
            #     p2_u = ( self.max_h - shape[1] ) // 2
            #     p2_d = self.max_h - shape[1] - p2_u
            # if shape[2] < self.max_w :
            #     p2_r = ( self.max_w - shape[2] ) // 2
            #     p2_l = self.max_w - shape[2] - p2_r

            # f_data  += [F.pad(data,  (p2_l, p2_r, p2_u, p2_d))]
            # f_label += [F.pad(label, (p2_l, p2_r, p2_u, p2_d))]
            
            f_data += [tr(data)]
            f_label += [tr(label * 255)] ### normalization for label

        f_data = torch.stack(f_data, dim=0)
        f_label = torch.stack(f_label, dim=0)

        return f_data, f_label

from matplotlib import pyplot as plt

if __name__ == '__main__':
    A2C_train_dset = HDataset('A2C', 'train')
    A2C_val_dset = HDataset('A2C', 'validation')
    A4C_test_dset = HDataset('A4C', 'test')

    A2C_data, A2C_label = A2C_train_dset.__getitem__(0)
    A2C_vdata, A2C_vlabel = A2C_val_dset.__getitem__(1)
    print(A2C_train_dset.__len__())
    print(A2C_train_dset.max_h, A2C_train_dset.max_w)
    print(A4C_test_dset.__len__())
    count = 0
    for x in A2C_data:
        print(x.shape)
        
        x = x.numpy()
        print(x.shape)
        plt.imshow(x, interpolation='nearest')
        plt.savefig(str(count)+'.png')
        count = count+1

        #plt.show()






