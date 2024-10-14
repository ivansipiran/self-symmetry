import torch
from utils import parser
from utils.config import *
from datasets.ShapeNetDataset import *
from data_handler import get_dataset
from symmetry import sel_dataloader, sel_model, sel_optimizer, sel_scheduler, unpack_elements
from screenshot import ScreenshotsPolyscope, NullScreenshot
import util
from torch.utils.data import DataLoader
from losses import chamfer_distance

def reflective_loss(N1, N2, N3, d2, device,args):
    Np1 = torch.squeeze(N1)
    Np2 = torch.squeeze(N2)
    Np3 = torch.squeeze(N3)

    Np1 = torch.nn.functional.normalize(Np1, dim=1)
    Np2 = torch.nn.functional.normalize(Np2, dim=1)
    Np3 = torch.nn.functional.normalize(Np3, dim=1)

    R1 = util.get_sym_matrix(Np1).to(device)
    R2 = util.get_sym_matrix(Np2).to(device)
    R3 = util.get_sym_matrix(Np3).to(device)

    output = d2.clone()
    output = output.transpose(2, 1)

    output1 = torch.bmm(output, R1)
    output2 = torch.bmm(output, R2)
    output3 = torch.bmm(output, R3)

    output1 = output1.transpose(2, 1)
    output2 = output2.transpose(2, 1)
    output3 = output3.transpose(2, 1)

    loss1 = chamfer_distance(output1, d2, mse=args.mse) + chamfer_distance(output2, d2, mse=args.mse) + chamfer_distance(output3, d2, mse=args.mse)

    M = torch.reshape(torch.cat((Np1, Np2, Np3),1), (d2.shape[0], 3, 3))
    MtM = torch.bmm(M, torch.transpose(M, 1, 2))

    I = torch.eye(3)
    I = I.reshape((1,3,3))
    I = I.repeat(d2.shape[0], 1, 1)

    I = torch.eye(M.size(-1), dtype=M.dtype, device=M.device)
    loss2 = torch.linalg.matrix_norm(torch.abs(MtM) - I, ord='fro')

    loss = (loss1 + loss2).mean()

    return loss

def rotational_loss(N1, N2, N3, device):
    Np1 = torch.squeeze(N1)
    Np2 = torch.squeeze(N2)
    Np3 = torch.squeeze(N3)

    Np1 = torch.nn.functional.normalize(Np1, dim=1)
    Np2 = torch.nn.functional.normalize(Np2, dim=1)
    Np3 = torch.nn.functional.normalize(Np3, dim=1)

    R11 = util.get_rot_matrix(Np1, 30).to(device)
    R12 = util.get_rot_matrix(Np1, 60).to(device)
    R13 = util.get_rot_matrix(Np1, 90).to(device)

    R21 = util.get_rot_matrix(Np2, 30).to(device)
    R22 = util.get_rot_matrix(Np2, 60).to(device)
    R23 = util.get_rot_matrix(Np2, 90).to(device)

    R31 = util.get_rot_matrix(Np3, 30).to(device)
    R32 = util.get_rot_matrix(Np3, 60).to(device)
    R33 = util.get_rot_matrix(Np3, 90).to(device)

    output = d2.clone()
    output = output.transpose(2, 1)

    output11 = torch.bmm(output, R11)
    output12 = torch.bmm(output, R12)
    output13 = torch.bmm(output, R13)

    output21 = torch.bmm(output, R21)
    output22 = torch.bmm(output, R22)
    output23 = torch.bmm(output, R23)

    output31 = torch.bmm(output, R31)
    output32 = torch.bmm(output, R32)
    output33 = torch.bmm(output, R33)

    output11 = output11.transpose(2, 1)
    output12 = output12.transpose(2, 1)
    output13 = output13.transpose(2, 1)

    output21 = output21.transpose(2, 1)
    output22 = output22.transpose(2, 1)
    output23 = output23.transpose(2, 1)

    output31 = output31.transpose(2, 1)
    output32 = output32.transpose(2, 1)
    output33 = output33.transpose(2, 1)

    loss1 = chamfer_distance(output11, d2, mse=args.mse) + chamfer_distance(output12, d2, mse=args.mse) + chamfer_distance(output13, d2, mse=args.mse)
    loss2 = chamfer_distance(output21, d2, mse=args.mse) + chamfer_distance(output22, d2, mse=args.mse) + chamfer_distance(output23, d2, mse=args.mse)
    loss3 = chamfer_distance(output31, d2, mse=args.mse) + chamfer_distance(output32, d2, mse=args.mse) + chamfer_distance(output33, d2, mse=args.mse)

    loss = loss1 + loss2 + loss3

    M = torch.reshape(torch.cat((Np1, Np2, Np3),1), (d2.shape[0], 3, 3))
    MtM = torch.bmm(M, torch.transpose(M, 1, 2))

    I = torch.eye(3)
    I = I.reshape((1,3,3))
    I = I.repeat(d2.shape[0], 1, 1)

    I = torch.eye(M.size(-1), dtype=M.dtype, device=M.device)
    loss2 = torch.linalg.matrix_norm(torch.abs(MtM) - I, ord='fro')

    loss = (loss + loss2).mean()

    return loss

def train_oneobject_produce_metric_planesym(d2, args, screenshot=NullScreenshot()):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    print(f'device: {device}')

    object = d2
    data_loader = get_dataset(args.sampling_mode)(object, device, args)

    model = sel_model(args, num_class=6)

    print(f'number of parameters: {util.n_params(model)}')
    model.to(device)
    optimizer = sel_optimizer(model.parameters(), args)
    scheduler = sel_scheduler(optimizer, args)

    model.train()
    train_loader = DataLoader(data_loader, num_workers=0,
                          batch_size=args.batch_size, shuffle=False, drop_last=False)

    epoch_number = 0
    args.dataset2 = "oneobject"
    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        output = None

        for i, d in enumerate(train_loader):

            tl_unpacked = unpack_elements(args, d)
            d2 = tl_unpacked["points"]

            d2 = d2.to(device) # 1 x dim x num_points
            
            model.train()
            optimizer.zero_grad()

            N1, N2, N3 = model(d2)

            loss = reflective_loss(N1, N2, N3, d2, device, args)

            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_number += 1

    for i, d in enumerate(train_loader):
        tl_unpacked = unpack_elements(args, d)
        d2 = tl_unpacked["points"]
        d2 = d2.to(device)
        d2 = d2[0,:,:]
        d2 = d2.unsqueeze(0)


        N1, N2, N3 = model(d2)
        
        Np1 = torch.squeeze(N1)
        Np2 = torch.squeeze(N2)
        Np3 = torch.squeeze(N3)
        Np1 = torch.nn.functional.normalize(Np1,dim=1)
        Np2 = torch.nn.functional.normalize(Np2,dim=1)
        Np3 = torch.nn.functional.normalize(Np3,dim=1)

        R1 = util.get_sym_matrix(Np1).to(device)
        R2 = util.get_sym_matrix(Np2).to(device)
        R3 = util.get_sym_matrix(Np3).to(device)

        x = d2.transpose(2, 1)
        x1 = torch.bmm(x, R1)
        x2 = torch.bmm(x, R2)
        x3 = torch.bmm(x, R3)

        x1 = x1.transpose(2, 1)
        x2 = x2.transpose(2, 1)
        x3 = x3.transpose(2, 1)


        chamfer_distance1 = chamfer_distance(x1, d2, mse=args.mse)
        chamfer_distance2 = chamfer_distance(x2, d2, mse=args.mse)
        chamfer_distance3 = chamfer_distance(x3, d2, mse=args.mse)

        minchamfer = torch.argsort(torch.Tensor([chamfer_distance1, chamfer_distance2, chamfer_distance3]))
        
        return Np1, Np2, Np3, minchamfer
        
def process_experiment(args, config):
    
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    
    dataset = ShapeNetDataset(root=config.dataset._base_.DATA_PATH, num_points=config.num_points, category=None)
    
    
    if config.screenshot.enable:
        if not (config.save_path / 'screenshots').exists():
            (config.save_path / 'screenshots').mkdir()
        Screenshot = ScreenshotsPolyscope(config.save_path / 'screenshots', enabled=config.screenshot.enable, extension='jpg')
    else:
        Screenshot = NullScreenshot()
    
    for i, (d2, category, name) in enumerate(dataset):
        print(f'-------------   OBJECT {i}-{category}-----------------------')

        d2 = d2.to(device)
        Screenshot.object_id = f'object{i}'

        train_output = train_oneobject_produce_metric_planesym(d2, config, Screenshot)
        N1 = train_output[0].cpu().detach().numpy()
        N2 = train_output[1].cpu().detach().numpy()
        N3 = train_output[2].cpu().detach().numpy()
        L = [N1, N2, N3]
        minchamfer = train_output[3]

        print(L[minchamfer[0]])
        print(L[minchamfer[1]])
        print(L[minchamfer[2]])
        break
        
        #Save symmetries
        if not os.path.exists(os.path.join(args.experiment_path, category)):
            os.makedirs(os.path.join(args.experiment_path, category))
        
        filename = os.path.join(args.experiment_path, category, name + '-sym.txt') 
        print(filename)
        with open(filename, 'wt') as f:
            f.write('3\n')
            
            for i in range(3):
                f.write('plane ')
                f.write(" ".join([str(x) for x in L[minchamfer[i]]]) + ' ')
                f.write('0 0 0\n')

def main():
    args = parser.get_args()
    args.use_gpu = torch.cuda.is_available()

    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    
    config = get_config(args)
    
    process_experiment(args, config)


if __name__=='__main__':
    print('aja')
    main()