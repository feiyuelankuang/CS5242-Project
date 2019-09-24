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
    def __init__(self, img_dir, labels):
        'Initialization'
        self.img_dir = img_dir
        self.labels = read_csv(labels)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img = np.load(self.img_dir+str(index)+'.npy')
        aspect_ratio = img.shape[0]/img.shape[1]
        img = Image.fromarray(img)
#        print(img)
#        print(img.shape)
        #print(self.labels)
        label = self.labels[index]      
        Transform = []
        Transform.append(T.Resize((256,256),Image.NEAREST))
#        Transform.append(T.Resize((int(300*aspect_ratio),300),Image.NEAREST))
#        Transform.append(T.RandomCrop((256,256)))
        Transform.append(T.RandomHorizontalFlip())
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        img = Transform(img)
#        print(label)
 #       print(img.shape)
        return img,torch.tensor(label)
