import torch

import torch.nn as nn
import torch.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from gan.datasets import ImageDataset

class Discriminator(object):
    def __init__(self, input_nc, cls_num=1):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, cls_num, 4, padding=1)]

        self.model = nn.Sequential(*model)
        self.init_all_optimizer()
        self.init_loss()
        self.init_dataset()
        self.cls_num = cls_num

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

    def init_all_optimizer(self, args):

        self.optimizer = self.init_optimizer(self.model.parameters(), args.lr)

    def init_dataset(self, args):
        # Dataset loader
        transforms_ = [transforms.Resize(int(args.size*1.12), Image.BICUBIC),
                       transforms.RandomCrop(args.size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))]

        self.dataloader = DataLoader(ImageDataset(args.dataroot, transforms_=transforms_, unaligned=True),
                                     batch_size=args.batchSize, shuffle=True, num_workers=args.n_cpu)

    def init_loss(self):

        self.criterion = nn.CrossEntropyLoss()

    def train_process(self, input, label):

        self.optimizer.zero_grad()

        x = self.forward(input)
        loss = self.criterion(x, label)

        loss.backward()

        self.optimizer.step()

        return loss

    def train(self):

        self.model.train()
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        frequency = self.args.frequency
        output_dir = self.args.output_dir

        for epoch in range(self.args.epoch, self.args.n_epochs):
            for i, batch in enumerate(self.dataloader):
                if self.cuda:
                    batch = to_cuda(batch)

                input = batch['A']
                B = batch['B']
                target_real = Tensor(A.shape[0]).fill_(1.0).unsqueeze(-1)
                target_fake = Tensor(A.shape[0]).fill_(0.0).unsqueeze(-1)

                loss_G, fake_A, fake_B = self.generator_process(
                    A, B, target_real)
                loss_D = self.discriminator_process(
                    A, B, fake_A, fake_B,  target_real, target_fake)
                self.logger.log({'loss_G': loss_G, 'loss_D': loss_D},
                                images={'real_A': A, 'real_B': B, 'fake_A': fake_A, 'fake_B': fake_B})

            if epoch % frequency == (frequency - 1):
                torch.save(self.G_A2B.state_dict(), output_dir +
                           '/{}_{}.pth'.format('G_A2B', epoch))

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