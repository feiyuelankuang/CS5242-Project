import torch
from torch.utils import data
import argparse
from dataset2 import Dataset
import torchvision
from utils import progress_bar
import torch.nn as nn
from resnet import resnet34,resnet18,resnet50
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import csv
from vgg import *
from cnn import *

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--batch_size', '--batch-size', default=32, type=int, help='initial learning rate')
parser.add_argument('--model', default='resnet34', type = str, help='model used')
args = parser.parse_args()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
# Parameters

max_epochs = 80

# Datasets
train_dir = 'data/train/'# IDs
train_label = 'data/train_kaggle.csv'
test_dir  = 'data/test/'# Labels
test_label = 'data/sample_solution.csv'
# Generators
train_set = Dataset(train_dir, train_label,'train')
val_set = Dataset(train_dir, train_label,'eval')
# Creating data indices for training and validation splits:
dataset_size = len(train_set)
#print(dataset_size)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))

np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_params = {'batch_size': args.batch_size,
          'num_workers': 2,
          'sampler':train_sampler
          }
val_params = {'batch_size': args.batch_size,
          'num_workers': 2,
          'sampler':val_sampler
          }

train_generator = torch.utils.data.DataLoader(train_set, **train_params)
#print(len(train_generator))
val_generator = torch.utils.data.DataLoader(val_set, **val_params)
#print(len(val_generator))

test_params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 2}
test_set = Dataset(test_dir, test_label)
test_generator = data.DataLoader(test_set, **test_params)

if args.model == 'resnet50':
    model = resnet50(pretrained=False,num_classes=2)
elif args.model == 'resnet34':
    model = resnet34(pretrained=False,num_classes=2)
elif args.model == 'resnet18':
    model = resnet18(pretrained=False,num_classes=2)
elif args.model == 'vgg11_bn':
    model = vgg11_bn(num_classes = 2)
elif args.model == 'MalwareNet':
    model = MalwareNet(in_feature = 102)



if use_cuda:
    model = model.to('cuda:0')

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr)  

statfile = open('training_'+args.model+'_lr'+str(args.lr)+'_batchsize'+str(args.batch_size)+'.txt', 'a+')

def decrease_learning_rate():
    """Decay the previous learning rate by 10"""
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 2

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0        
 #   training_setting = 'batchsize=%d | epoch=%d | lr=%.1e ' % (batchsize, epoch, optimizer.param_groups[0]['lr'])  

    for batch_idx, (inputs, targets) in enumerate(train_generator):
        if use_cuda:
            inputs, targets = inputs.to(device,dtype=torch.float), targets.to(device)
#        print(targets.shape)
        optimizer.zero_grad()
        #print(inputs)
        outputs = model(inputs)
 #       print(outputs.shape)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
#        print(loss.item())
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
                
#        print(predicted.shape)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum() 
        if batch_idx == 0:
            y_true = targets.cpu().numpy()
            y_score = predicted.cpu().numpy()
        else:
            y_true = np.concatenate((y_true,targets.cpu().numpy()))
            y_score = np.concatenate((y_score,predicted.cpu().numpy()))

        progress_bar(batch_idx, len(train_generator), 'Loss: %.3f | Acc: %.3f'
            % (train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total)))

    acc = 100.*(float)(correct)/(float)(total)
    auc = roc_auc_score(y_true, y_score)
    print('auc score is: ', auc)
    statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | AUC: %.3f' \
                % (epoch, train_loss/(batch_idx+1), acc, correct, total, auc)
    statfile.write(statstr+'\n')

best_auc = 0

def val(epoch):
    print('\nEpoch: %d' % epoch)
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    global best_auc
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_generator):
            if use_cuda:
                inputs, targets = inputs.to(device,dtype=torch.float), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
#        print(loss.item())
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
                
#        print(predicted.shape)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum() 
            if batch_idx == 0:
                y_true = targets.cpu().numpy()
                y_score = predicted.cpu().numpy()
            else:
                y_true = np.concatenate((y_true,targets.cpu().numpy()))
                y_score = np.concatenate((y_score,predicted.cpu().numpy()))

            progress_bar(batch_idx, len(val_generator), 'Loss: %.3f | Acc: %.3f'
                % (val_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total)))

    acc = 100.*(float)(correct)/(float)(total)
    auc = roc_auc_score(y_true, y_score)
    print('auc score is: ', auc)
    if auc > best_auc:
        best_auc = auc
        if epoch > 100:
            checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_auc' : auc
            }
            torch.save(checkpoint, 'checkpoint/'+args.model+'_best_epoch'+'.t7')
            test(epoch)
    statstr = 'Validating: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | AUC: %.3f' \
                % (epoch, val_loss/(batch_idx+1), acc, correct, total, auc)
    statfile.write(statstr+'\n')


start_epoch = 1

def test(epoch):
    model.eval()
    print(len(test_generator))
    with torch.no_grad():
        with open('result/epoch_'+str(epoch)+'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['Id','Predicted'])
            for idx, (inputs, targets) in enumerate(test_generator):
                if use_cuda:
                    inputs, targets = inputs.to(device,dtype=torch.float), targets.to(device)

                outputs = model(inputs)
                #print(outputs.data)                
                _, predicted = torch.max(outputs.data, 1)
                #print(predicted)
                writer.writerow([int(idx),int(predicted.data)])
        csvFile.close()   

for epoch in range(start_epoch, start_epoch+400):
    if epoch % 40 == 0:
        decrease_learning_rate()       
    train(epoch)
    val(epoch)






    # Validation
#    with torch.set_grad_enabled(False):
#        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
#            local_batch, local_labels = local_batch.to(device), local_labels.to(device)