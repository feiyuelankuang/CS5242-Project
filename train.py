import torch
from torch.utils import data
import argparse
from dataset import Dataset
import torchvision
from utils import progress_bar
import torch.nn as nn
from resnet import resnet34
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
args = parser.parse_args()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
# Parameters
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 2}
val_params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1}


max_epochs = 100

# Datasets
train_dir = 'data/train/'# IDs
train_label = 'data/train_kaggle.csv'
test_dir  = 'data/test/'# Labels
test_label = 'data/sample_solution.csv'
# Generators
training_set = Dataset(train_dir, train_label)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(test_dir, test_label)
validation_generator = data.DataLoader(validation_set, **val_params)

model = resnet34(pretrained=False,num_classes=2)
if use_cuda:
    model = model.to('cuda:0')

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr)  

statfile = open('training_status.txt', 'a+')

def decrease_learning_rate():
    """Decay the previous learning rate by 10"""
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 10

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

        
 #   training_setting = 'batchsize=%d | epoch=%d | lr=%.1e ' % (batchsize, epoch, optimizer.param_groups[0]['lr'])  

    for batch_idx, (inputs, targets) in enumerate(training_generator):
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
#        print(targets.shape)
        optimizer.zero_grad()
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

        progress_bar(batch_idx, len(training_generator), 'Loss: %.3f | Acc: %.3f'
            % (train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total)))

    acc = 100.*(float)(correct)/(float)(total)
    auc = roc_auc_score(y_true, y_scores)
    print('auc score is: ', auc)
    statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | AUC: %.3f' \
                % (epoch, train_loss/(batch_idx+1), acc, correct, total, best_acc)
    statfile.write(statstr+'\n')

start_epoch = 0

for epoch in range(start_epoch, start_epoch+100):
    if epoch==40 or epoch==60 or epoch==80:
        decrease_learning_rate()       
    train(epoch)

def test(filename):
    with torch.no_grad():
        with open(filename, 'w') as csvFile:
            writer = csv.writer(csvFile)
            for idx, (inputs, targets) in enumerate(validation_generator):
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                writer.writerow([idx,predicted])
        csvFile.close()   

test('result.csv')



    # Validation
#    with torch.set_grad_enabled(False):
#        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
#            local_batch, local_labels = local_batch.to(device), local_labels.to(device)