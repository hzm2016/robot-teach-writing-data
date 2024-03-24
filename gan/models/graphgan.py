from typing import runtime_checkable
import pandas as pd
import torch
import os
import cv2
import torch.nn.functional as F
import argparse
import numpy as np

from torch_geometric.nn import GATConv, GraphConv, global_max_pool, global_mean_pool, XConv, PointConv, radius_graph, fps, global_max_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from sklearn.metrics import pairwise_distances
from datasets import SequentialImageDataset
from utils import ReplayBuffer, Logger, to_cuda
from torch.utils.data import DataLoader
from torch.nn import Linear
from basegan import GAN
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def visulize_points(points, num, name):

    num = np.cumsum(num)
    img_canvas = np.full((128, 128), 255, np.uint8)
    points = points.astype(int)
    for idx, value in enumerate(num):

        if idx == 0:
            st = 0
        else:
            st = num[idx - 1]

        ed = num[idx]

        sub_points = points[st:ed]
        c = (0, 0, 0)
        for i, _ in enumerate(sub_points[:-1]):
            cv2.line(img_canvas, (sub_points[i][0], sub_points[i][1]),
                     (sub_points[i+1][0], sub_points[i+1][1]), c, 2)

    sub_points = points[ed:]
    c = (0, 0, 0)
    for i, _ in enumerate(sub_points[:-1]):
        cv2.line(img_canvas, (sub_points[i][0], sub_points[i][1]),
                    (sub_points[i+1][0], sub_points[i+1][1]), c, 2)
    # cv2.imwrite(name, img_canvas)

    return img_canvas


class GraphGeneratorPointNet(torch.nn.Module):

    def __init__(self, out_channels=2) -> None:
        super(GraphGeneratorPointNet, self).__init__()

        nn = Seq(Lin(2, 64), ReLU(), Lin(64, 64))
        self.conv1 = PointConv(local_nn=nn)

        nn = Seq(Lin(66, 128), ReLU(), Lin(128, 128))
        self.conv2 = PointConv(local_nn=nn)

        nn = Seq(Lin(130, 256), ReLU(), Lin(256, 256))
        self.conv3 = PointConv(local_nn=nn)

        self.lin1 = Lin(256, 256)
        self.lin2 = Lin(256, 256)
        self.lin3 = Lin(256, 128)

        nn = Seq(Lin(258, 256), ReLU(), Lin(256, 256))
        self.conv4 = PointConv(local_nn=nn)

        nn = Seq(Lin(258, 128), ReLU(), Lin(128, 128))
        self.conv5 = PointConv(local_nn=nn)
        
        self.lin6 = Lin(256, 128)
        self.lin4 = Lin(128, 64)
        self.lin5 = Lin(64, out_channels)

    def forward_prior(self, pos, edge_index):

        radius = 0.2
        batch = torch.zeros(pos.shape[0]).long().cuda()
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv1(None, pos, edge_index))

        radius = 0.4
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv2(x, pos, edge_index))

        radius = 0.6
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv3(x, pos, edge_index))

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin3(x))

        return x

    def forward_output(self, pos, edge_index, prior):

        radius = 1
        batch = torch.zeros(pos.shape[0]).long().cuda()
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv1(None, pos, edge_index))

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv2(x, pos, edge_index))

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv3(x, pos, edge_index))

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)

        radius = 1
        batch_size = x.shape[0]
        x = torch.cat([x, prior.repeat(batch_size, 1)], dim=1)

        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv4(x, pos, edge_index))

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv5(x, pos, edge_index))

        x = F.relu(self.lin4(x))
        x = self.lin5(x)

        return x

    def forward_output_single(self, x):
        
        x = F.relu(self.lin6(x)) 
        x = F.relu(self.lin4(x))
        x = self.lin5(x)

        return x

    def forward(self, o, m, c, edge_index_o, edge_index_m, edge_index_c):

        o_feature = self.forward_prior(o, edge_index_o)
        m_feature = self.forward_prior(m, edge_index_m)

        prior_feature = torch.cat([o_feature, m_feature], dim=1)
        # prior_feature = o_feature - m_feature
        # generated_result = self.forward_output(c, edge_index_c, prior_feature)
        generated_result = self.forward_output_single(prior_feature)
        
        return generated_result + c, generated_result


class GraphDiscriminatorPointNet(torch.nn.Module):

    def __init__(self, cls_num=1) -> None:
        super(GraphDiscriminatorPointNet, self).__init__()

        nn = Seq(Lin(2, 64), ReLU(), Lin(64, 64))
        self.conv1 = PointConv(local_nn=nn)

        nn = Seq(Lin(66, 128), ReLU(), Lin(128, 128))
        self.conv2 = PointConv(local_nn=nn)

        nn = Seq(Lin(130, 256), ReLU(), Lin(256, 256))
        self.conv3 = PointConv(local_nn=nn)

        self.lin1 = Lin(256, 256)
        self.lin2 = Lin(256, 256)
        self.lin3 = Lin(256, cls_num)

    def forward(self, pos, edge_index):

        radius = 1
        batch = torch.zeros(pos.shape[0]).long().cuda()
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv1(None, pos, edge_index))

        # idx = fps(pos, batch, ratio=0.5)
        # x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv2(x, pos, edge_index))

        # idx = fps(pos, batch, ratio=0.25)
        # x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv3(x, pos, edge_index))

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)

        return x


class GraphGenerator(torch.nn.Module):

    def __init__(self, out_channels=16) -> None:
        super(GraphGenerator, self).__init__()
        self.conv1 = GraphConv(2, 8)
        self.conv2 = GraphConv(8, out_channels)

        self.conv3 = GraphConv(2, out_channels)
        self.conv4 = GraphConv(out_channels*3, 2)

    def forward_prior(self, x, edge_index):

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward_output(self, x, edge_index, prior):

        batch_size = x.shape[0]
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv3(x, edge_index))

        x = torch.cat([x, prior.repeat(batch_size, 1)], dim=1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, edge_index)

        return torch.sigmoid(x)

    def forward(self, o, m, c, edge_index_o, edge_index_m, edge_index_c):

        dummy_batch = torch.zeros(1).long().cuda()

        o = self.forward_prior(o, edge_index_o)
        o_feature = global_max_pool(o, dummy_batch)

        m = self.forward_prior(m, edge_index_m)
        m_feature = global_max_pool(m, dummy_batch)

        prior_feature = torch.cat([o_feature, m_feature], dim=1)

        generate_result = self.forward_output(c, edge_index_c, prior_feature)

        return generate_result + c, c


class GraphDiscriminator(torch.nn.Module):

    def __init__(self, cls_num=1) -> None:
        super(GraphDiscriminator, self).__init__()
        self.conv1 = GraphConv(2, 16)
        self.conv2 = GraphConv(16, 32)

        self.lin1 = Linear(32, 16)
        self.lin2 = Linear(16, 8)
        self.lin3 = Linear(8, cls_num)

    def forward(self, x, edge_index):

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)

        batch = torch.zeros(1).long().cuda()
        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin3(x))

        return x
        # return F.log_softmax(x, dim=-1)


class GraphGAN(GAN):

    def __init__(self, args, mode='train') -> None:
        super().__init__()
        self.cuda = args.cuda
        self.args = args
        self.init_networks(args)
        if mode == 'train':
            self.init_all_optimizer(args)
            self.init_dataset(args)
            self.init_loss(args)
            self.logger = Logger(args.n_epochs, len(self.dataloader))
        if self.cuda:
            self.to_cuda()

    def to_cuda(self,):

        self.G.cuda()
        self.D.cuda()

    def init_networks(self, args, out_channels=32):

        self.G = GraphGeneratorPointNet()
        self.D = GraphDiscriminatorPointNet()

    def init_all_optimizer(self, args):

        self.optimizer_G = self.init_optimizer(self.G.parameters(), args.lr)
        self.optimizer_D = self.init_optimizer(self.D.parameters(), args.lr)
        self.G_scheduler = self.init_scheduler(
            self.optimizer_G, args.n_epochs, args.epoch, args.decay_epoch)
        self.D_scheduler = self.init_scheduler(
            self.optimizer_D, args.n_epochs, args.epoch, args.decay_epoch)

    def init_loss(self, args):

        self.criterion_GAN = torch.nn.MSELoss() 
        self.criterion_DIS = torch.nn.MSELoss(reduction='mean')
        self.criterion_REC = torch.nn.MSELoss()

        self.fake_buffer = ReplayBuffer()

    def init_dataset(self, args):

        self.dataloader = DataLoader(SequentialImageDataset(args.dataroot),
                                     batch_size=args.batchSize, shuffle=True, num_workers=args.n_cpu)

    def generator_process(self, o_points, m_points, c_points,
                          edge_index_o, edge_index_m, edge_index_c, target_real, dist_dis=None, l_points=None):

        self.optimizer_G.zero_grad()
        fake_c, shift = self.G(o_points, m_points, c_points,
                               edge_index_o, edge_index_m, edge_index_c)

        fake_batch = torch.cat([m_points, fake_c], dim=0)

        fake_edge = pairwise_distances(fake_batch.detach().cpu().numpy()) < 0.1
        fake_edge = torch.from_numpy(fake_edge.astype(int)).long().cuda()
        fake_edge = fake_edge.nonzero().t()

        if l_points is not None:
            loss_DIS = self.criterion_DIS(fake_c, l_points) * 128
            loss_G = loss_DIS  
        elif dist_dis is not None:
            loss_DIS = self.criterion_DIS(shift, dist_dis.repeat(shift.shape[0],1)) * 128
            pred_fake = self.D(fake_batch, fake_edge)
            loss_GAN = self.criterion_GAN(pred_fake, target_real)
            loss_G = loss_GAN + loss_DIS
        else:
            pred_fake = self.D(fake_batch, fake_edge)
            loss_GAN = self.criterion_GAN(pred_fake, target_real)
            loss_G = loss_GAN

        loss_REC = self.criterion_REC(fake_c, c_points) * 128

        loss_G.backward()
        self.optimizer_G.step()

        return fake_batch, loss_G, loss_REC

    def discriminator_process(self, fake_batch, real_batch, target_real, target_fake):

        self.optimizer_D.zero_grad()

        real_edge = pairwise_distances(real_batch.cpu().numpy()) < 0.1
        real_edge = torch.from_numpy(real_edge.astype(int)).long().cuda()
        real_edge = real_edge.nonzero().t()

        fake_edge = pairwise_distances(fake_batch.detach().cpu().numpy()) < 0.1
        fake_edge = torch.from_numpy(fake_edge.astype(int)).long().cuda()
        fake_edge = fake_edge.nonzero().t()

        pred_real = self.D(real_batch, real_edge)
        loss_D_real = self.criterion_GAN(pred_real, target_real)

        fake = self.fake_buffer.push_and_pop(fake_batch)
        pred_fake = self.D(fake_batch.detach(), fake_edge)
        loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

        loss_D = (loss_D_real + loss_D_fake)*0.5

        loss_D.backward()

        self.optimizer_D.step()

        return loss_D, loss_D_real, loss_D_fake

    def train(self, args):

        self.G.train()
        self.D.train()

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        frequency = self.args.frequency
        output_dir = self.args.output_dir

        for epoch in range(self.args.epoch, self.args.n_epochs):
            for i, batch in enumerate(self.dataloader):
                if self.cuda:
                    batch = to_cuda(batch)

                o_points = batch['o_points'].float().squeeze(0)
                m_points = batch['m_points'].float().squeeze(0)
                c_points = batch['c_points'].float().squeeze(0)
                l_points = batch['l_points'].float().squeeze(0) if 'l_points' in batch.keys() else None
                dist_dis = batch['dist_dis']
                real_batch = torch.cat([o_points, c_points], dim=0)

                edge_index_o = batch['edge_index_o']
                edge_index_m = batch['edge_index_m']
                edge_index_c = batch['edge_index_c']

                edge_index_o = edge_index_o[0].nonzero().t()
                edge_index_m = edge_index_m[0].nonzero().t()
                edge_index_c = edge_index_c[0].nonzero().t()

                target_real = Tensor(1).fill_(1.).unsqueeze(-1)
                target_fake = Tensor(1).fill_(0.).unsqueeze(-1)
                # target_real = torch.tensor([1]).cuda()
                # target_fake = torch.tensor([0]).cuda()

                fake_batch, loss_G, loss_REC = self.generator_process(
                    o_points, m_points, c_points, edge_index_o, edge_index_m, edge_index_c, target_real, dist_dis, l_points)

                loss_D, loss_D_real, loss_D_fake = self.discriminator_process(
                    fake_batch, real_batch, target_real, target_fake)

                self.logger.log({'loss_G': loss_G, 'loss_REC': loss_REC, 'loss_D': loss_D, 'loss_D_real': loss_D_real, 'loss_D_fake': loss_D_fake},
                                images=None)

            if epoch % frequency == (frequency - 1):
                torch.save(self.G.state_dict(), output_dir +
                           '/{}_{}.pth'.format('G', epoch))

            self.G_scheduler.step()
            self.D_scheduler.step()

    def test(self, args):

        self.G.eval()
        self.D.eval()

        test_dataloader = DataLoader(SequentialImageDataset(args.dataroot),
                                     batch_size=args.batchSize, shuffle=False, num_workers=args.n_cpu)

        os.makedirs(self.args.output_dir + '/fake', exist_ok=True)
        os.makedirs(self.args.output_dir + '/real', exist_ok=True)
        os.makedirs(self.args.output_dir + '/comb', exist_ok=True)

        for i, batch in tqdm(enumerate(test_dataloader)):
            if self.cuda:
                batch = to_cuda(batch)

            o_points = batch['o_points'].float().squeeze(0)
            m_points = batch['m_points'].float().squeeze(0)
            c_points = batch['c_points'].float().squeeze(0)
            l_points = batch['l_points'].float().squeeze(0)
            num = batch['num'].cpu().numpy()
            m_num = batch['m_num'].cpu().numpy()

            edge_index_o = batch['edge_index_o']
            edge_index_m = batch['edge_index_m']
            edge_index_c = batch['edge_index_c']

            edge_index_o = edge_index_o[0].nonzero().t()
            edge_index_m = edge_index_m[0].nonzero().t()
            edge_index_c = edge_index_c[0].nonzero().t()

            fake_c, shift = self.G(o_points, m_points, c_points,
                                   edge_index_o, edge_index_m, edge_index_c)

            fake = torch.cat([m_points, fake_c], dim=0)
            fake = fake.detach().cpu().numpy() * 128

            real = torch.cat([o_points, c_points], dim=0)
            real = real.cpu().numpy() * 128

            fake = visulize_points(fake, m_num, self.args.output_dir +
                            '/fake/%04d.png' % (i+1))
            real = visulize_points(real, num, self.args.output_dir +
                            '/real/%04d.png' % (i+1))
            
            comb = np.zeros((128, 256))
            comb[:128,:128] = real
            comb[:128,128:] = fake

            cv2.imwrite(self.args.output_dir +
                            '/comb/%04d.png' % (i+1), comb)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--model', type=str,
                        default='CycleGAN', help='type of models')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='data/seq/',
                        help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--input_nc', type=int, default=1,
                        help='number of channels of input font_data')
    parser.add_argument('--output_nc', type=int, default=1,
                        help='number of channels of output font_data')
    parser.add_argument('--cuda', action='store_true',
                        help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=0,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--frequency', type=int, default=20,
                        help='frequency of saving trained model')
    parser.add_argument('--output_dir', type=str,
                        default='./output', help='place to output result')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    model = GraphGAN(args)

    key_pairs = {
        'G': '/home/cunjun/Robot-Teaching-Assiantant/gan/graphgan/G_59.pth'
    }
    model.train(args)
    # model.load_networks(key_pairs)
    model.test(args)
