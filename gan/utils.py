import random
import time
import datetime
import itertools
import sys

from torch.autograd import Variable
import torch
# from visdom import Visdom
import torchvision.transforms as transforms
import numpy as np
from gan.nn.modules import Generator, StyleGanGenerator
from gan.nn.modules import Discriminator
from PIL import Image
from gan.datasets import ImageDataset, SequentialImageDataset
from torch.utils.data import DataLoader


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        # self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        # for image_name, tensor in images.items():
        #     if image_name not in self.image_windows:
        #         self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
        #     else:
        #         self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                # if loss_name not in self.loss_windows:
                #     self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                #                                                     opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                # else:
                #     self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)


def train_d_net(optimizer, d_net, real, fake, fake_buffer, criterion, target_real, target_fake):

    optimizer.zero_grad()

    # Real loss
    pred_real = d_net(real)
    loss_D_real = criterion(pred_real, target_real)

    # Fake loss
    fake = fake_buffer.push_and_pop(fake)
    pred_fake = d_net(fake.detach())
    loss_D_fake = criterion(pred_fake, target_fake)

    # Total loss
    loss_D = (loss_D_real + loss_D_fake)*0.5
    loss_D.backward()

    optimizer.step()

    return loss_D


def get_cyc_loss(real, g_net, criterion, target, const=1):

    fake = g_net(real)
    loss = criterion(fake, target)*const
    return fake, loss


def get_cyc_loss_con(real, condition, g_net, criterion, target, const=1):

    fake = g_net(real, condition)
    loss = criterion(fake, target)*const
    return fake, loss


def get_gan_loss(real, g_net, d_net, criterion, target, const=1):

    fake = g_net(real)
    pred = d_net(fake)
    loss = criterion(pred, target)*const
    return fake, loss


def get_gan_loss_con(real, condition, g_net, d_net, criterion, target, const=1):

    fake = g_net(real, condition)
    pred = d_net(fake)
    loss = criterion(pred, target)*const
    return fake, loss

def to_cuda(batch):

    if isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = batch[k].cuda()
        return batch
    elif isinstance(batch, list):
        for i in batch:
            i = i.cuda()
        return batch
    else:
        raise NotImplementedError

def init_networks(opt):

    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netG_S2P = Generator(opt.output_nc, opt.input_nc, True)
    netG_P2S = Generator(opt.output_nc, opt.input_nc, True)

    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)
    netD_S = Discriminator(opt.output_nc)
    netD_P = Discriminator(opt.output_nc, 5)

    if opt.cuda:
        return netG_A2B.cuda(), netG_B2A.cuda(), netG_S2P.cuda(), netG_P2S.cuda(), netD_A.cuda(), netD_B.cuda(), netD_P.cuda(), netD_S.cuda()
    else:
        return netG_A2B, netG_B2A, netG_S2P, netG_P2S, netD_A, netD_B, netD_P, netD_S


def init_loss(opt, cls_num=4):

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_GAN_CLS = torch.nn.CrossEntropyLoss()

    return criterion_GAN_CLS, criterion_GAN, criterion_cycle, criterion_identity


def init_optimizer(opt, netG_A2B, netG_B2A, netG_S2P, netG_P2S, netD_A, netD_B, netD_P, netD_S):

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters(), netG_S2P.parameters(), netG_P2S.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(
        netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(
        netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_P = torch.optim.Adam(
        netD_P.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_S = torch.optim.Adam(
        netD_S.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_P = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_P, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_S = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_S, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    return optimizer_G, optimizer_D_A, optimizer_D_B, optimizer_D_P, optimizer_D_S, lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B, lr_scheduler_D_P, lr_scheduler_D_S


def init_dataset(opt):
    # Dataset loader
    transforms_ = [transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5,), (0.5,))]

    if opt.sequential:
        dataloader = DataLoader(SequentialImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                                batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    else:
        dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                                batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    return dataloader