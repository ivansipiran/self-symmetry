import torch
#from torch_cluster import knn

class Loss():
    def __init__(self, args):
        self.logger = args.logger
        self.loss = []
        self.args = args

    #FOR ONE (CURRENT) ITERATION
    def log(self):
        self.logger.logloss(self.current_iter_data())
        pass

    def current_iter_data(self):
        return ""

    def val(self):
        pass

#Chamfer, determinant - 1, sym matrix transpose
class CDSLoss(Loss):
    def __init__(self, args):
        super().__init__(args)
        self.loss1 = torch.Tensor([-1])
        self.loss2 = torch.Tensor([-1])
        self.loss3 = torch.Tensor([-1])
        self.loss_history1 = []
        self.loss_history2 = []
        self.loss_history3 = []

    def loss(self, d2, output, sym_matrix, args):
        self.loss1 = chamfer_distance(output, d2, mse=args.mse)
        # R = torch.squeeze(sym_matrix)
        R = sym_matrix
        # loss2 = torch.pow(torch.det(R) + 1,2)
        # loss2 = torch.pow(torch.det(R) + 1,2).sum()
        self.loss2 = torch.pow(torch.det(R) + 1, 2).sum()  # batches sum()
        RtR = torch.matmul(torch.transpose(R, 2, 1), R)
        I = torch.eye(R.size(-1), dtype=R.dtype, device=R.device)
        # loss3 = torch.nn.functional.l1_loss(RtR, I) # seems to be working with batches, even as I doesn't have the batch dimension
        self.loss3 = torch.mean(torch.abs(RtR - I), dim=[0, 1, 2])
        loss = self.loss1 + self.loss2 + self.loss3

        self.loss_history1.append(self.loss1)
        self.loss_history2.append(self.loss2)
        self.loss_history3.append(self.loss3)

        return loss

    def current_iter_data(self):
        #if self.valid():
        #    return f"Loss1: {util.show_truncated(self.loss1.item(), 6)}; Loss2: {util.show_truncated(self.loss2.item(), 6)}; Loss3: {util.show_truncated(self.loss3.item(), 6)}')"
        return (self.loss1.item(), self.loss2.item(), self.loss3.item())

    # noinspection PyTypeChecker
    def valid(self):
        return not torch.any(torch.cat([self.loss1,self.loss2,self.loss3]) == torch.Tensor([-1, -1, -1]))



def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor, mse=False, normalize1=False, normalize2=False):
    pc1, pc2 = pc1.transpose(1, 2), pc2.transpose(1, 2)

    # Batch X N1 X N2 X 3
    distance_matrix = (pc1[:, :, None, :3] - pc2[:, None, :, :3]).norm(dim=-1)
    if mse:
        distance_matrix = distance_matrix ** 2
    d12, _ = distance_matrix.min(dim=2)
    d21, _ = distance_matrix.min(dim=1)
    if normalize1:
        n12 = pc1.shape[1]
        n21 = pc2.shape[1]
        return d12.mean(dim=-1).sum()/n12 + d21.mean(dim=-1).sum()/n21
    if normalize2:
        maxd12, _ = d12.max(dim=1)
        maxd21, _ = d21.max(dim=1)
        n12 = maxd12*pc1.shape[1]
        n21 = maxd21*pc2.shape[1]
        return d12.mean(dim=-1).sum()/n12 + d21.mean(dim=-1).sum()/n21
    return d12.mean(dim=-1).sum() + d21.mean(dim=-1).sum()

"""
def chamfer_distance_cluster(pc1: torch.Tensor, pc2: torch.Tensor):
    def get_batch(data):
        batch_size, N, _ = data.shape  # (batch_size, num_points, 3/6)
        pos = data.view(batch_size * N, -1)
        batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
        for i in range(batch_size): batch[i] = i
        batch = batch.view(-1)
        return pos, batch

    batch_size = pc1.shape[0]
    pc1, pc2 = pc1.transpose(1, 2), pc2.transpose(1, 2)
    pc1, pc2 = pc1.contiguous(), pc2.contiguous()
    pc1, batch1 = get_batch(pc1)
    pc2, batch2 = get_batch(pc2)
    nn2_1 = knn(pc1, pc2, 1, batch_x=batch1, batch_y=batch2)[1]
    nn1_2 = knn(pc2, pc1, 1, batch_x=batch2, batch_y=batch1)[1]
    d21 = (pc1[nn2_1] - pc2)[:, :3].norm(dim=-1).mean() * batch_size
    d12 = (pc2[nn1_2] - pc1)[:, :3].norm(dim=-1).mean() * batch_size
    return d21 + d12
"""