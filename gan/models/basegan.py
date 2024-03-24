import torch
import itertools
from gan.utils import LambdaLR
import torch.nn.functional as F


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

class GAN(object):

    def __init__(self) -> None:
        super().__init__()

    
    def load_networks(self, key_pairs):

        for key, value in key_pairs.items():
            getattr(self, key).load_state_dict(
                {k.replace('module.', ''): v for k, v in torch.load(value).items()})

    def generator(self):

        raise NotImplementedError

    def discriminator(self, d_net, criterion, real, fake, fake_buffer, target_fake, target_real):
        # Real loss
        pred_real = d_net(real)
        loss_D_real = criterion(pred_real, target_real)

        # Fake loss
        fake = fake_buffer.push_and_pop(fake)
        pred_fake = d_net(fake.detach())
        loss_D_fake = criterion(pred_fake, target_fake)

        # Total loss
        loss_D = (loss_D_real + loss_D_fake)*0.5

        return loss_D
    
    def discriminator_logistic(self, d_net, real, fake, fake_buffer):
        # Real loss
        pred_real = d_net(real)

        fake = fake_buffer.push_and_pop(fake)
        pred_fake = d_net(fake.detach())

        # Total loss
        loss_D = d_logistic_loss(pred_real, pred_fake)

        return loss_D

    def train(self, dataset):
        raise NotImplementedError

    @staticmethod
    def init_optimizer(module, lr, beta=(0.5, 0.999)):

        optimizer = torch.optim.Adam(module,
                                         lr=lr, betas=(0.5, 0.999))

        return optimizer
        
    @staticmethod
    def init_scheduler(optimizer, n_epochs, start_epoch, decay_epoch):

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)

        return lr_scheduler

    @staticmethod
    def save_model(self, output_dir, epoch, model, name, frequency=20):
        
        if epoch % frequency == (frequency-1):
                torch.save(model.state_dict(), output_dir +
                           '/{}_{}.pth'.format(name, epoch))
