import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout

from torch.nn import Identity as BatchNorm1d

from torch.nn import init
import torch.nn as nn

class SymmetryNetwork(nn.Module):
    def __init__(self):
        super(SymmetryNetwork, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        #iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)
        #if x.is_cuda:
        #    iden = iden.cuda()
        #x = x + iden
        x = x.view(-1, 3, 3)
        return x

class SymmetryNetwork2(nn.Module):
    def __init__(self):
        super(SymmetryNetwork2, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        
        self.head1 = nn.Linear(9, 3)
        self.head2 = nn.Linear(9, 3)
        self.head3 = nn.Linear(9, 3)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)

        return x1, x2, x3


class SymmetryNetwork14V(nn.Module):
    def __init__(self, device):
        super(SymmetryNetwork14V, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32)

        self.heads = [nn.Linear(32, 3).to(device) for i in range(14)]

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x_14v = [self.heads[i](x) for i in range(14)]

        return x_14v

""" class PointNet2Generator(torch.nn.Module):
    '''
    ref:
        - https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py
        - https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet++.py
    '''
    def __init__(self, device, args):
        super(PointNet2Generator, self).__init__()
        nsamples = 32

        # SA1
        sa1_sample_ratio = 0.7
        sa1_radius = 0.025
        sa1_max_num_neighbours = nsamples
        sa1_mlp = make_mlp(3 + 3, [32, 32, 64])
        self.sa1_module = PointNet2SAModule(sa1_sample_ratio,
                                            sa1_radius, sa1_max_num_neighbours, sa1_mlp)

        # SA2
        sa2_sample_ratio = 0.7
        sa2_radius = 0.05
        sa2_max_num_neighbours = nsamples
        sa2_mlp = make_mlp(64+3, [64, 64, 64, 128])
        self.sa2_module = PointNet2SAModule(sa2_sample_ratio,
                                            sa2_radius, sa2_max_num_neighbours, sa2_mlp)

        # SA3
        sa3_sample_ratio = 0.7
        sa3_radius = 0.1
        sa3_max_num_neighbours = nsamples
        sa3_mlp = make_mlp(128 + 3, [128, 128, 128, 256])
        self.sa3_module = PointNet2SAModule(sa3_sample_ratio,
                                            sa3_radius, sa3_max_num_neighbours, sa3_mlp)

        # SA4
        sa4_sample_ratio = 0.7
        sa4_radius = 0.2
        sa4_max_num_neighbours = nsamples
        sa4_mlp = make_mlp(256 + 3, [256, 256, 256, 512])
        self.sa4_module = PointNet2SAModule(sa4_sample_ratio,
                                            sa4_radius, sa4_max_num_neighbours, sa4_mlp)

        knn_num = 3

        # FP3, reverse of sa3
        fp4_knn_num = knn_num
        fp4_mlp = make_mlp(512 + 256 + 3, [512, 256, 256, 256])
        self.fp4_module = PointNet2FPModule(fp4_knn_num, fp4_mlp)

        # FP3, reverse of sa3
        fp3_knn_num = knn_num
        fp3_mlp = make_mlp(256+128+3, [256, 256, 256, 256])
        self.fp3_module = PointNet2FPModule(fp3_knn_num, fp3_mlp)

        # FP2, reverse of sa2
        fp2_knn_num = knn_num
        fp2_mlp = make_mlp(256+64+3, [256, 256])
        self.fp2_module = PointNet2FPModule(fp2_knn_num, fp2_mlp)

        # FP1, reverse of sa1
        fp1_knn_num = knn_num
        fp1_mlp = make_mlp(256+3+3, [256, 128, 128, 128])
        self.fp1_module = PointNet2FPModule(fp1_knn_num, fp1_mlp)

        self.fc = torch.nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            torch.nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=1, bias=False),
            nn.Conv1d(32, 16, kernel_size=1, bias=False),
            nn.Conv1d(16, 3, kernel_size=1, bias=False),
        )


    def forward(self, data):
        '''
        data: a batch of input, torch.Tensor or torch_geometric.data.Data type
            - torch.Tensor: (batch_size, 3, num_points), as common batch input
            - torch_geometric.data.Data, as torch_geometric batch input:
                data.x: (batch_size * ~num_points, C), batch nodes/points feature,
                    ~num_points means each sample can have different number of points/nodes
                data.pos: (batch_size * ~num_points, 3)
                data.batch: (batch_size * ~num_points,), a column vector of graph/pointcloud
                    idendifiers for all nodes of all graphs/pointclouds in the batch. See
                    pytorch_gemometric documentation for more information
        '''
        input_points = data.clone()

        # Convert to torch_geometric.data.Data type
        data = data.transpose(1, 2).contiguous()
        batch_size, N, _ = data.shape  # (batch_size, num_points, 3/6)
        pos = data.view(batch_size*N, -1)
        batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
        for i in range(batch_size): batch[i] = i
        batch = batch.view(-1)

        data = Data()
        data.x, data.pos, data.batch = pos, pos[:, :3].detach(), batch

        if not hasattr(data, 'x'): data.x = None
        data_in = data.x, data.pos, data.batch

        sa1_out = self.sa1_module(data_in)
        sa2_out = self.sa2_module(sa1_out)
        sa3_out = self.sa3_module(sa2_out)
        sa4_out = self.sa4_module(sa3_out)

        fp4_out = self.fp4_module(sa4_out, sa3_out)
        fp3_out = self.fp3_module(fp4_out, sa2_out)
        fp2_out = self.fp2_module(fp3_out, sa1_out)
        fp1_out = self.fp1_module(fp2_out, data_in)

        fp1_out_x, fp1_out_pos, fp1_out_batch = fp1_out
        x = self.fc(fp1_out_x.transpose(1, 0).unsqueeze(0)).transpose(1, 2).squeeze(0)
        return x.view(batch_size, N, 3).transpose(2, 1) + input_points

    def initialize_params(self, val=0.2):
        self.std = val
        for p in self.parameters():
            torch.nn.init.uniform_(p.data, -val, val) """
