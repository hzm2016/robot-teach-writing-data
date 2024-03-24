from genericpath import exists
import torch
import itertools
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from gan.nn.modules import Generator, Discriminator
from gan.nn import StyledGenerator2, StyleDiscriminator
from gan.utils import ReplayBuffer, to_cuda, Logger
from gan.datasets import ImageDataset
from .basegan import GAN

import torch.nn.functional as F
from torch.utils.data import DataLoader

accum = 0.5 ** (32 / (10 * 1000))


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


class StyleCycleGAN(GAN):

    def __init__(self, args, mode='train') -> None:
        super().__init__()

        self.styled = True
        self.args = args
        if mode == 'inference':
            self.init_network_inference(args)
            self.cuda = args.get('CUDA')
        else:
            self.init_networks(args)
            self.cuda = args.cuda
            self.init_all_optimizer(args)
            self.init_dataset(args)
            self.init_loss(args)
            self.logger = Logger(args.n_epochs, len(self.dataloader))
        if self.cuda:
            self.to_cuda()

    def init_networks(self, args):

        if self.styled:
            self.G_A2B = torch.nn.DataParallel(StyledGenerator2(128, 512))
            self.G_B2A = torch.nn.DataParallel(StyledGenerator2(128, 512))

            self.G_A2B_EMA = StyledGenerator2(128, 512)
            self.G_B2A_EMA = StyledGenerator2(128, 512)

            self.D_A = torch.nn.DataParallel(StyleDiscriminator(128))
            self.D_B = torch.nn.DataParallel(StyleDiscriminator(128))

            # self.D_A = Discriminator(1)
            # self.D_B = Discriminator(1)

        else:
            self.G_A2B = Generator(1, 1)
            self.G_B2A = Generator(1, 1)

            self.G_A2B_EMA = Generator(1, 1)
            self.G_B2A_EMA = Generator(1, 1)

            self.D_A = Discriminator(1)
            self.D_B = Discriminator(1)

    def init_network_inference(self, args):

        self.G_A2B = StyledGenerator2(128, 512)
        self.G_B2A = StyledGenerator2(128, 512)

        self.D_A = StyleDiscriminator(128)
        self.D_B = StyleDiscriminator(128)

    def to_cuda(self):
        to_cuda([self.G_A2B, self.G_B2A, self.D_A,
                self.D_B, self.G_A2B_EMA, self.G_B2A_EMA])

    def init_all_optimizer(self, args):

        self.optimizer_G = self.init_optimizer(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()), args.lr)
        self.optimizer_D = self.init_optimizer(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()), args.lr)
        self.G_scheduler = self.init_scheduler(
            self.optimizer_G, args.n_epochs, args.epoch, args.decay_epoch)
        self.D_scheduler = self.init_scheduler(
            self.optimizer_D, args.n_epochs, args.epoch, args.decay_epoch)

    def init_dataset(self, args):
        # Dataset loader
        transforms_ = [transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))]

        self.dataloader = DataLoader(ImageDataset(args.dataroot, transforms_=transforms_, unaligned=True),
                                     batch_size=args.batchSize, shuffle=True, num_workers=args.n_cpu)

    def init_loss(self, args):

        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def generator_process(self, A, B, target_real):

        self.optimizer_G.zero_grad()

        requires_grad(self.G_A2B, True)
        requires_grad(self.G_A2B, True)
        requires_grad(self.D_A, False)
        requires_grad(self.D_B, False)

        # GAN loss
        fake_B = self.G_A2B(A)
        pred_fake = self.D_B(fake_B)
        # loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)
        loss_GAN_A2B = g_nonsaturating_loss(pred_fake)

        fake_A = self.G_B2A(B)
        pred_fake = self.D_A(fake_A)
        # loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)
        loss_GAN_B2A = g_nonsaturating_loss(pred_fake)

        # Cycle loss
        recovered_A = self.G_B2A(fake_B)
        loss_cycle_ABA = self.criterion_cycle(recovered_A, A)*5.0

        recovered_B = self.G_A2B(fake_A)
        loss_cycle_BAB = self.criterion_cycle(recovered_B, B)*5.0

        # Sum all losses
        G_loss = loss_GAN_A2B + loss_GAN_B2A
        C_loss = loss_cycle_ABA + loss_cycle_BAB
        loss_G = G_loss + C_loss

        loss_G.backward(retain_graph=True)
        self.optimizer_G.step()

        accumulate(self.G_B2A_EMA, self.G_B2A.module, accum)
        accumulate(self.G_A2B_EMA, self.G_A2B.module, accum)

        return loss_G, fake_A, fake_B, G_loss, C_loss

    def discriminator_process(self, A, B, fake_A, fake_B, target_real, target_fake):

        self.optimizer_D.zero_grad()

        requires_grad(self.G_A2B, False)
        requires_grad(self.G_A2B, False)
        requires_grad(self.D_A, True)
        requires_grad(self.D_B, True)

        # D_loss_A = self.discriminator(
        #     self.D_A, self.criterion_GAN, A, fake_A, self.fake_A_buffer, target_fake, target_real)
        # D_loss_B = self.discriminator(
        #     self.D_B, self.criterion_GAN, B, fake_B, self.fake_B_buffer, target_fake, target_real)
        D_loss_A = self.discriminator_logistic(
            self.D_A, A, fake_A, self.fake_A_buffer)
        D_loss_B = self.discriminator_logistic(
            self.D_B, B, fake_B, self.fake_B_buffer)

        loss_D = D_loss_A + D_loss_B

        loss_D.backward()

        self.optimizer_D.step()

        return loss_D, D_loss_A, D_loss_B

    def train(self):

        self.G_B2A.train()
        self.G_A2B.train()
        self.D_A.train()
        self.D_B.train()

        self.G_B2A_EMA.train()
        self.G_A2B_EMA.train()

        accumulate(self.G_B2A_EMA, self.G_B2A.module, 0)
        accumulate(self.G_A2B_EMA, self.G_A2B.module, 0)

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        frequency = self.args.frequency
        output_dir = self.args.output_dir

        vis_sample = None
        os.makedirs(self.args.output_dir + '/B', exist_ok=True)

        for epoch in range(self.args.epoch, self.args.n_epochs):
            for i, batch in enumerate(self.dataloader):
                if self.cuda:
                    batch = to_cuda(batch)

                A = batch['A']
                B = batch['B']
                if vis_sample is None:
                    vis_sample = A[:1]
                    sample = 0.5*(vis_sample.data + 1.0)
                    save_image(sample, self.args.output_dir +
                               '/B/asample.png'.format(epoch, i))
                target_real = Tensor(A.shape[0]).fill_(1.0).unsqueeze(-1)
                target_fake = Tensor(A.shape[0]).fill_(0.0).unsqueeze(-1)

                loss_G, fake_A, fake_B, G_loss, C_loss = self.generator_process(
                    A, B, target_real)
                loss_D, D_loss_A, D_loss_B = self.discriminator_process(
                    A, B, fake_A, fake_B, target_real, target_fake)
                self.logger.log({'loss_G': loss_G, 'G_loss': G_loss, 'C_loss': C_loss, 'loss_D': loss_D, 'D_loss_A': D_loss_A, 'D_loss_B': D_loss_B},
                                images={'real_A': A, 'real_B': B})

                if i % 100 == 0:
                    fake_B = 0.5*(self.G_A2B_EMA(vis_sample).data + 1.0)
                    save_image(fake_B, self.args.output_dir +
                               '/B/epoch_{}_{}.png'.format(epoch, i))

            if epoch % frequency == (frequency - 1):
                torch.save(self.G_A2B_EMA.state_dict(), output_dir +
                           '/{}_{}.pth'.format('G_A2B', epoch))
                torch.save(self.G_B2A_EMA.state_dict(), output_dir +
                           '/{}_{}.pth'.format('G_B2A', epoch))

            self.G_scheduler.step()
            self.D_scheduler.step()

    def test(self,):

        self.G_A2B.eval()
        self.G_B2A.eval()

        transforms_ = [transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))]

        test_dataloader = DataLoader(ImageDataset(self.args.dataroot, transforms_=transforms_, unaligned=False, mode='train'),
                                     batch_size=self.args.batchSize, shuffle=False, num_workers=self.args.n_cpu)

        os.makedirs(self.args.output_dir + '/A', exist_ok=True)
        os.makedirs(self.args.output_dir + '/B', exist_ok=True)

        for i, batch in tqdm(enumerate(test_dataloader)):
            # Set model input
            A = batch['A']
            B = batch['B']

            if self.cuda:
                A = A.cuda()
                B = B.cuda()

            # Generate output
            fake_B = 0.5*(self.G_A2B(A).data + 1.0)
            fake_A = 0.5*(self.G_B2A(B).data + 1.0)

            # Save image files
            save_image(fake_A, self.args.output_dir + '/A/%04d.png' % (i+1))
            save_image(fake_B, self.args.output_dir + '/B/%04d.png' % (i+1))
