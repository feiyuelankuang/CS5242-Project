import torch
from torch.utils import data
import csv
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


def read_csv(csv_file):
    label_list=[]
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row[1].isalpha():                
                label_list.append(int(float(row[1])>0.5))
    return label_list
#print(read_csv('data/sample_solution.csv'))

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, img_dir, labels, mode = 'eval'):
        'Initialization'
        self.img_dir = img_dir
        self.labels = read_csv(labels)
        self.mode = mode

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img = np.load(self.img_dir+str(index)+'.npy')
#        print(img)
        x = img.shape[0]
        y = img.shape[1]
#        print(x,y)
        img = Image.fromarray(img)

#        print(img)
#        print(img.shape)
        #print(self.labels)
        label = self.labels[index]      
        Transform = []
        if self.mode == 'train':
#        Transform.append(T.Resize((256,256),Image.NEAREST))
            if x > y:
                Transform.append(T.Resize((int(256*x/y),256),Image.NEAREST))
            else:
                Transform.append(T.Resize((256,int(256*y/x)),Image.NEAREST))

            Transform.append(T.RandomCrop((224,224)))
            Transform.append(T.RandomHorizontalFlip())
            #Transform.append(T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02))
        else:
            Transform.append(T.Resize((224,224),Image.NEAREST))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        img = Transform(img)
#        print(label)
#        print(img)
        return img,torch.tensor(label)
