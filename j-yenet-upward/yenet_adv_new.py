# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:25:14 2019

@author: sun
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import time


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

#from filters import all_normalized_hpf_list
from srm_filter_kernel import all_normalized_hpf_list
#from MPNCOV.python import MPNCOV

IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
EPOCHS = 280

LR = 0.002
WEIGHT_DECAY = 5e-4


TRAIN_FILE_COUNT = 14000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [130, 230]
TMP = 230

OUTPUT_PATH = Path(__file__).stem
os.makedirs(OUTPUT_PATH, exist_ok=True)

class TLU(nn.Module):
  def __init__(self, threshold):
    super(TLU, self).__init__()

    self.threshold = threshold

  def forward(self, input):
    output = torch.clamp(input, min=-self.threshold, max=self.threshold)

    return output

class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
      all_hpf_list_5x5.append(hpf_item)

    hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)


    self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=True)
    self.hpf.weight = hpf_weight

    #Truncation, threshold = 5
    self.tlu = TLU(31.0)


  def forward(self, input):

    output = self.hpf(input)
    output = self.tlu(output)


    return output

class block1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block1, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()
        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                #nn.ReLU(),
                #nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )

    def forward(self,input):
        output=self.basic(input)
        
        output+=input
       
        output=self.relu(output)
        return output
    
class block2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block2, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()
        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(),
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(outchannel),
                #nn.ReLU(),
                #nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )
        self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(outchannel),
                )

    def forward(self,input):
        output=self.basic(input)
        
        output+=self.shortcut(input)
        
        output=self.relu(output)
        return output

class block3(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block3, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()
        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, inchannel//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(inchannel//2),
                nn.ReLU(),
                nn.Conv2d(inchannel//2, inchannel//2, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(inchannel//2),
                nn.ReLU(),
                nn.Conv2d(inchannel//2, outchannel, kernel_size=1, padding=0),
                nn.BatchNorm2d(outchannel),
                #nn.ReLU(),
                #nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )
        self.shortcut=nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0),
                nn.BatchNorm2d(outchannel),
                )

    def forward(self,input):
        output=self.basic(input)
        
        output+=self.shortcut(input)
        
        output=self.relu(output)
        return output

class block4(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block4, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()
        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, inchannel//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(inchannel//2),
                nn.ReLU(),
                nn.Conv2d(inchannel//2, inchannel//2, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(inchannel//2),
                nn.ReLU(),
                nn.Conv2d(inchannel//2, outchannel, kernel_size=1, padding=0),
                nn.BatchNorm2d(outchannel),
                #nn.ReLU(),
                #nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )
        self.shortcut=nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0),
                nn.BatchNorm2d(outchannel),
                )

    def forward(self,input):
        output=self.basic(input)
        
        output+=self.shortcut(input)
        
        output=self.relu(output)
        return output    

class block5(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block5, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()
        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, inchannel//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(inchannel//2),
                nn.ReLU(),
                nn.Conv2d(inchannel//2, inchannel//2, kernel_size=3, padding=1,stride=3),
                nn.BatchNorm2d(inchannel//2),
                nn.ReLU(),
                nn.Conv2d(inchannel//2, outchannel, kernel_size=1, padding=0),
                nn.BatchNorm2d(outchannel),
                #nn.ReLU(),
                #nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )
        self.shortcut=nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=3,padding=1),
                nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0),
                nn.BatchNorm2d(outchannel),
                )
        self.pool=nn.AvgPool2d(kernel_size=11,stride=1)
        
    def forward(self,input):
        output=self.basic(input)
        
        output+=self.shortcut(input)
       
        output=self.relu(output)
        output=self.pool(output)
        return output

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.group1 = HPF()

    self.group2 = nn.Sequential(
      nn.Conv2d(30, 30, kernel_size=3, padding=1),
      nn.BatchNorm2d(30),
      nn.ReLU(),

      nn.Conv2d(30, 30, kernel_size=3, padding=1),
      nn.BatchNorm2d(30),
      nn.ReLU(),

      nn.Conv2d(30, 30, kernel_size=3, padding=1),
      nn.BatchNorm2d(30),
      nn.ReLU()      
    )

    self.group3 = block1(30,30)
    self.group4 = block1(30,30)
    self.group5 = block2(30,60)
    self.group6 = block3(60,64)
    self.group7 = block4(64,128)
    self.group8 = block5(128,256)
    
    #self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)
    self.fc1 = nn.Linear(1 * 1 * 256, 2)

  def forward(self, input):
    output = input

    output = self.group1(output)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)
    output = self.group5(output)
    output = self.group6(output)
    output = self.group7(output)
    output = self.group8(output)

#    output = MPNCOV.CovpoolLayer(output)
#    output = MPNCOV.SqrtmLayer(output, 5)
#    output = MPNCOV.TriuvecLayer(output)

    output = output.view(output.size(0), -1)
    output = self.fc1(output)

    return output

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch):
  batch_time = AverageMeter() #ONE EPOCH TRAIN TIME
  data_time = AverageMeter()
  losses = AverageMeter()

  model.train()

  end = time.time()

  for i, sample in enumerate(train_loader):

    data_time.update(time.time() - end) 

    data, label = sample['data'], sample['label']

    shape = list(data.size())
    data = data.reshape(shape[0] * shape[1], *shape[2:])
    label = label.reshape(-1)


    # fig = plt.figure(figsize=(10, 10))

    # rows = 2
    # coloums = 6

    # for i in range(1, coloums + 1):
    #   fig.add_subplot(rows, coloums, i)

    #   plt.imshow(np.squeeze(data[i + 10]))

    # for i in range(1, coloums + 1):
    #   fig.add_subplot(rows, coloums, coloums + i)

    #   plt.imshow(np.squeeze(sc[i + 10]))

    # plt.show()

    # return

    data, label = data.to(device), label.to(device)

    optimizer.zero_grad()

    end = time.time()

    output = model(data) #FP


    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)

    losses.update(loss.item(), data.size(0))

    loss.backward()      #BP
    optimizer.step()

    batch_time.update(time.time() - end) #BATCH TIME = BATCH BP+FP
    end = time.time()

    if i % TRAIN_PRINT_FREQUENCY == 0:
      # logging.info('Epoch: [{}][{}/{}] \t Loss {:.6f}'.format(epoch, i, len(train_loader), loss.item()))

      logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def adjust_bn_stats(model, device, train_loader):
  model.train()

  with torch.no_grad():
    for sample in train_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP):
  model.eval()

  test_loss = 0
  correct = 0

  with torch.no_grad():
    for sample in eval_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(label.view_as(pred)).sum().item()

  accuracy = correct / (len(eval_loader.dataset) * 2)
  all_state = {
      'original_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'epoch': epoch
    }
  torch.save(all_state, PARAMS_PATH1)
  if accuracy > best_acc and epoch > TMP:
    best_acc = accuracy
    all_state = {
      'original_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'epoch': epoch
    }
    torch.save(all_state, PARAMS_PATH)
  
  logging.info('-' * 8)
  logging.info('Eval accuracy: {:.4f}'.format(accuracy))
  logging.info('Best accuracy:{:.4f}'.format(best_acc))   
  logging.info('-' * 8)
  return best_acc

def initWeights(module):
  if type(module) == nn.Conv2d:
    if module.weight.requires_grad:
      nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

      # nn.init.xavier_uniform_(module.weight.data)
      nn.init.constant_(module.bias.data, val=0)
    else:
      module.weight.requires_grad = True
      nn.init.constant_(module.bias.data, val=0)

  if type(module) == nn.Linear:
    nn.init.normal_(module.weight.data, mean=0, std=0.01)
    nn.init.constant_(module.bias.data, val=0)


class AugData():
  def __call__(self, sample):
    data, label = sample['data'], sample['label']

    rot = random.randint(0,3)


    data = np.rot90(data, rot, axes=[1, 2]).copy()


    if random.random() < 0.5:
      data = np.flip(data, axis=2).copy()

    new_sample = {'data': data, 'label': label}

    return new_sample


class ToTensor():
  def __call__(self, sample):
    data, label = sample['data'], sample['label']

    data = np.expand_dims(data, axis=1)
    data = data.astype(np.float32)
    # data = data / 255.0

    new_sample = {
      'data': torch.from_numpy(data),
      'label': torch.from_numpy(label).long(),
    }

    return new_sample


class MyDataset(Dataset):
  def __init__(self, index_path, BB_COVER_DIR, BB_STEGO_DIR, transform=None):
    self.index_list = np.load(index_path)
    self.transform = transform

    self.cover_path = BB_COVER_DIR + '/{}.mat'
    self.stego_path = BB_STEGO_DIR + '/{}.mat'

    # self.bows_cover_path = BOWS_COVER_DIR + '/{}.mat'
    # self.bows_stego_path = BOWS_STEGO_DIR + '/{}.mat'

  def __len__(self):
    return self.index_list.shape[0]

  def __getitem__(self, idx):
    file_index = self.index_list[idx]

  # if file_index <= 10000:
    cover_path = self.cover_path.format(file_index)
    stego_path = self.stego_path.format(file_index)
    # else:
    #   cover_path = self.bows_cover_path.format(file_index - 10000)
    #   stego_path = self.bows_stego_path.format(file_index - 10000)


    #cover_data = cv2.imread(cover_path, -1)
    #stego_data = cv2.imread(stego_path, -1)
    cover_data = sio.loadmat(cover_path)['img']
    stego_data = sio.loadmat(stego_path)['img']


    data = np.stack([cover_data, stego_data])
    label = np.array([0, 1], dtype='int32')

    sample = {'data': data, 'label': label}

    if self.transform:
      sample = self.transform(sample)

    return sample

def setLogger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def main(args):

#  setLogger(LOG_PATH, mode='w')

#  Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
  statePath = args.statePath

  device = torch.device("cuda")

  kwargs = {'num_workers': 1, 'pin_memory': True}

  train_transform = transforms.Compose([
    AugData(),
    ToTensor()
  ])

  eval_transform = transforms.Compose([
    ToTensor()
  ])

  DATASET_INDEX = args.DATASET_INDEX
  STEGANOGRAPHY = args.STEGANOGRAPHY
  EMBEDDING_RATE = args.EMBEDDING_RATE
  ADVERSARIAL_RATE = args.ADVERSARIAL_RATE
  JPEG_QUALITY = args.JPEG_QUALITY

  if STEGANOGRAPHY == 'dmmr':
    base_dir = '/data1/lml/watermarking'
  else:
    base_dir = '/data/lml/watermarking'

  BB_COVER_DIR = '{}/BB-cover-resample-256-jpeg-{}-none-round'.format(base_dir, JPEG_QUALITY)
  # BB_STEGO_DIR = '{}/BB-cover-resample-256-jpeg-{}-{}-payload-{}-none-round'.format(base_dir, JPEG_QUALITY, STEGANOGRAPHY, EMBEDDING_RATE)
  BB_STEGO_DIR = '{}/BB-cover-resample-256-jpeg-{}-{}-upward-new-payload-{}-use-dcts-13-ae1-{}-none-round'.format(base_dir, JPEG_QUALITY, STEGANOGRAPHY, EMBEDDING_RATE, ADVERSARIAL_RATE)

  #RESTORE_FILTER_PATH = 'tmp/filter_32_5_norm_jpeg_{}_non_rounded_params.pt'.format(JPEG_QUALITY)

  TRAIN_INDEX_PATH = 'index_list/{}/train_list_14k.npy'.format(DATASET_INDEX)
  # TRAIN_INDEX_PATH = 'index_list/bossbase_train_index.npy'
  VALID_INDEX_PATH = 'index_list/{}/valid_list_1k.npy'.format(DATASET_INDEX)
  TEST_INDEX_PATH = 'index_list/{}/test_list_5k.npy'.format(DATASET_INDEX)
  
  LOAD_RATE = float(EMBEDDING_RATE) + 0.02
  # LOAD_RATE = round(LOAD_RATE, 1)
  
  global LR
  global DECAY_EPOCH
  global EPOCHS
  global TMP

  if EMBEDDING_RATE != '0.10': 
    LR = 0.001
    DECAY_EPOCH = [50, 80]
    EPOCHS = 110
    TMP = 80

  if JPEG_QUALITY=='95': 
    LR = 0.001
    DECAY_EPOCH = [50, 80]
    EPOCHS = 110
    TMP = 80
    
  if LOAD_RATE == 0.5 and JPEG_QUALITY=='95': 
    LR = 0.001 
    DECAY_EPOCH = [130, 230]
    EPOCHS = 280
    TMP = 230

  PARAMS_NAME = '{}-{}-{}-params_{}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX,  JPEG_QUALITY)
  LOG_NAME = '{}-{}-{}-model_log_{}'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX,  JPEG_QUALITY)
  
  PARAMS_NAME1 = '{}-{}-{}-process-params_{}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX,JPEG_QUALITY)
  
  PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
  PARAMS_PATH1 = os.path.join(OUTPUT_PATH, PARAMS_NAME1)
  LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

  # #transfer learning 
  PARAMS_INIT_NAME = '{}-{:.2f}-{}-params_{}.pt'.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, JPEG_QUALITY)
  
  # if LOAD_RATE == 0.5 and JPEG_QUALITY == '95':
  #   PARAMS_INIT_NAME = '{}-{}-{}-params_{}.pt'.format(STEGANOGRAPHY, '0.4', DATASET_INDEX, '75')
  #   #PARAMS_INIT_NAME = '{}-{}-{}-params-{}-lr={}-{}-{}-{}_{}.pt'.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, TIMES, '0.005', 80,140,180,JPEG_QUALITY)
   
  PARAMS_INIT_PATH = os.path.join(OUTPUT_PATH, PARAMS_INIT_NAME)
  print(PARAMS_INIT_PATH)

  setLogger(LOG_PATH, mode='w')

  Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
  
  train_dataset = MyDataset(TRAIN_INDEX_PATH, BB_COVER_DIR, BB_STEGO_DIR, train_transform)
  valid_dataset = MyDataset(VALID_INDEX_PATH, BB_COVER_DIR, BB_STEGO_DIR, eval_transform)
  test_dataset = MyDataset(TEST_INDEX_PATH, BB_COVER_DIR, BB_STEGO_DIR, eval_transform)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


  model = Net().to(device)  
  model.apply(initWeights) 
  params = model.parameters()

  hpf_params = list(map(id, model.group1.parameters()))
  res_params = filter(lambda p: id(p) not in hpf_params, model.parameters())
       
  param_groups = [{'params': res_params, 'weight_decay': WEIGHT_DECAY},
                    {'params': model.group1.parameters()}]

  optimizer = optim.Adamax(param_groups, lr=LR)

  # optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9)

  if statePath:
    logging.info('-' * 8)
    logging.info('Load state_dict in {}'.format(statePath))
    logging.info('-' * 8)

    all_state = torch.load(statePath)

    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    epoch = all_state['epoch']

    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    startEpoch = epoch + 1

  else:
    startEpoch = 1
  
  if EMBEDDING_RATE != '0.10' : 
    all_state = torch.load(PARAMS_INIT_PATH)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)
  
  # if LOAD_RATE == 0.5 and JPEG_QUALITY=='95':
  #   all_state = torch.load(PARAMS_INIT_PATH)
  #   original_state = all_state['original_state']
  #   model.load_state_dict(original_state)
      
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
  best_acc = 0.0
  for epoch in range(startEpoch, EPOCHS + 1):
    scheduler.step()

    train(model, device, train_loader, optimizer, epoch)

    if epoch % EVAL_PRINT_FREQUENCY == 0:
      adjust_bn_stats(model, device, train_loader)
      best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP)

  logging.info('\nTest set accuracy: \n')

   #load best parmater to test    
  all_state = torch.load(PARAMS_PATH)
  original_state = all_state['original_state']
  optimizer_state = all_state['optimizer_state']
  model.load_state_dict(original_state)
  optimizer.load_state_dict(optimizer_state)

  adjust_bn_stats(model, device, train_loader)
  evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP)


def myParseArgs():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-i',
    '--DATASET_INDEX',
    help='Path for loading dataset',
    type=str,
    default='1'
  )

  parser.add_argument(
    '-alg',
    '--STEGANOGRAPHY',
    help='embedding_algorithm',
    type=str,
    choices=['dmmr','gmas'],
    default = 'dmmr'
  )

  parser.add_argument(
    '-rate',
    '--EMBEDDING_RATE',
    help='embedding_rate',
    type=str,
    # choices=['0.1', '0.2', '0.3', '0.4'],
    default = '0.10'
  )
  parser.add_argument(
    '-p1',
    '--ADVERSARIAL_RATE',
    help='embedding_rate',
    type=str,
    # choices=['0.1', '0.2', '0.3', '0.4'],
    default = '0.70'
  )
  parser.add_argument(
    '-quality',
    '--JPEG_QUALITY',
    help='JPEG_QUALITY',
    type=str,
    choices=['75', '65'],
    default = '65'
  )

  parser.add_argument(
    '-g',
    '--gpuNum',
    help='Determine which gpu to use',
    type=str,
    choices=['0', '1', '2', '3'],
    required=True
  )

  parser.add_argument(
    '-l',
    '--statePath',
    help='Path for loading model state',
    type=str,
    default=''
  )

  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = myParseArgs()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
  main(args)