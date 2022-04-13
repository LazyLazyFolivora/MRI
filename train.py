import gc

import numpy as np
import torch
import random
import os
import torchcontrib.optim
import torchvision
from torch.autograd.grad_mode import F
from torch.utils.data import dataloader
from torchvision.transforms import transforms
from torch.optim import SGD
from torch.backends import cudnn
from torch.autograd import Variable
from MRI import MRI
from torchcontrib.optim import SWA
import time
import gc
import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.optim.lr_scheduler import CosineAnnealingLR
from Config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
# 是否打印训练进度
print_train_process = config.print_train_process

# 是否打印测试进度
print_test_process = config.print_test_process

# 是否保存最好的模型
save_best_model = config.save_best_model


class Best(object):
    def __init__(self):
        self.best = 0

    def get_best(self):
        return self.best

    def set_best(self, _best):
        self.best = _best


class Static(object):
    __init = None

    def __new__(cls, value):
        if cls.__init is None:
            cls.__init = object.__new__(cls)
            cls.__init.value = value
        return './accuracy/' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


best_test = Best()
file_name = Static(True)

log_interval = config.log_interval
num_epoch = config.epoch
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# num_classes
num_classes = config.num_classes

trainset = torchvision.datasets.ImageFolder(root=config.train_path,
                                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True,
                                          num_workers=config.num_workers, drop_last=True)

testset = torchvision.datasets.ImageFolder(root=config.test_path,
                                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=True,
                                         num_workers=config.num_workers)

model = MRI()

# 损失函数
# loss = CenterLoss()
loss = nn.CrossEntropyLoss()
# 优化器
sgd = SGD(model.parameters(), lr=0.005, momentum=0.5)
scheduler = CosineAnnealingLR(sgd, T_max=100)
model = nn.DataParallel(model)
cudnn.benchmark = True
model = model.cuda(device)

opt = torchcontrib.optim.SWA(sgd, swa_lr=0.025)
if config.continue_train:
    if os.path.exists(config.model_dict_path):
        model.load_state_dict(torch.load(config.model_dict_path, map_location=device), strict=False)
    else:
        raise Exception("无已保存的文件!")

def train(epoch):
    model.train()
    for batch, (data, target) in enumerate(trainloader):
        if torch.cuda.is_available():
            data = Variable(data).cuda()
            target = Variable(target).cuda()
        else:
            data = Variable(data)
            target = Variable(target)
        sgd.zero_grad()
        output, att, l = model(data, target)
        l.backward()
        sgd.step()
        if batch % log_interval == 0:
            if print_train_process:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch * len(data), len(trainloader.dataset),
                           100. * batch / len(trainloader), l.item()))
                print('each layer attention:', att)
                torch.cuda.empty_cache()
    if epoch > 10:
        opt.step()
    else:
        scheduler.step()


def test(_best):
    model.eval()
    test_loss = 0
    correct = 0
    for batch, (data, target) in enumerate(testloader):
        with torch.no_grad():
            if print_test_process:
                print(batch)
            if torch.cuda.is_available():
                data = Variable(data).cuda()
                target = Variable(target).cuda()
            else:
                data = Variable(data)
                target = Variable(target)

            output, att, l = model(data, target)
            test_loss += l
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    t = '\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset))
    file = open(r'' + file_name.replace(':', '-') + '.txt', 'a+')
    file.writelines(t)
    file.close()
    if save_best_model:
        if _best.get_best() < correct:
            best_test.set_best(correct)
        else:
            return
    torch.save(model.state_dict(), config.model_dict_path)
    torch.save(model, config.model_path)


if __name__ == '__main__':
    for epoch in range(1, num_epoch + 1):
        train(epoch)
        test(best_test)

    opt.swap_swa_sgd()
    test()
