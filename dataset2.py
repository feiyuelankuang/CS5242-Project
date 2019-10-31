import torch
from torch.utils import data
import csv
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

def normalized(x):
#    print(np.min(x),np.max(x))
    for i in range(x.shape[1]):
        z = x[:,i]
        if np.max(z)-np.min(z)>0:
            x[:,i]=(z-np.min(z))/(np.max(z)-np.min(z))
    return 255*x.astype(float)

def padding(x):
    target = np.zeros((1024,102))
    target[:x.shape[0],] = x
    return target

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
        #img = img * 255.0
        img = normalized(img)
        img = padding(img)
        img = np.reshape(img,(32,32,102))
#        print(img)

#        print(x,y)

#        print(img)
#        print(img.shape)
        #print(self.labels)
        label = self.labels[index]      
        Transform = []
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        img = Transform(img)
#        print(label)
#        print(img)
        return img,torch.tensor(label)
