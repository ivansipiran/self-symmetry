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

def train_oneobject_produce_metric_planesym(d2, args, screenshot=NullScreenshot()):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    print(f'device: {device}')

    #logger_obj = CDSLossLogger(args)

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
            Np1 = torch.squeeze(N1)
            Np2 = torch.squeeze(N2)
            Np3 = torch.squeeze(N3)
            Np1 = torch.nn.functional.normalize(Np1, dim=0)
            Np2 = torch.nn.functional.normalize(Np2, dim=0)
            Np3 = torch.nn.functional.normalize(Np3, dim=0)

            R1 = util.point_and_normal_to_matrix(Np1[None, :, None], device).to(device)
            R2 = util.point_and_normal_to_matrix(Np2[None, :, None], device).to(device)
            R3 = util.point_and_normal_to_matrix(Np3[None, :, None], device).to(device)

            output = d2.clone()
            output = output.transpose(2, 1)

            output1 = torch.bmm(output, R1)
            output2 = torch.bmm(output, R2)
            output3 = torch.bmm(output, R3)

            output1 = output1.transpose(2, 1)
            output2 = output2.transpose(2, 1)
            output3 = output3.transpose(2, 1)

            loss1 = chamfer_distance(output1, d2, mse=args.mse) + chamfer_distance(output2, d2,
                                                                                   mse=args.mse) + chamfer_distance(
                output3, d2, mse=args.mse)
            M = torch.cat((Np1.unsqueeze(1), Np2.unsqueeze(1), Np2.unsqueeze(1)), 1)

            MtM = torch.matmul(torch.transpose(M, 0, 1), M)
            I = torch.eye(M.size(-1), dtype=M.dtype, device=M.device)
            loss2 = torch.linalg.matrix_norm(torch.abs(MtM) - I, ord='fro')
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_number += 1

    for i, d in enumerate(train_loader):
        tl_unpacked = unpack_elements(args, d)
        d2 = tl_unpacked["points"]
        d2 = d2.to(device)

        N1, N2, N3 = model(d2)
        Np1 = torch.squeeze(N1)
        Np2 = torch.squeeze(N2)
        Np3 = torch.squeeze(N3)
        Np1 = torch.nn.functional.normalize(Np1,dim=0)
        Np2 = torch.nn.functional.normalize(Np2,dim=0)
        Np3 = torch.nn.functional.normalize(Np3,dim=0)

        R1 = util.point_and_normal_to_matrix(Np1[None, :, None], device).to(device)
        R2 = util.point_and_normal_to_matrix(Np2[None, :, None], device).to(device)
        R3 = util.point_and_normal_to_matrix(Np3[None, :, None], device).to(device)

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

        #minchamfer = torch.argmin(torch.Tensor([chamfer_distance1, chamfer_distance2, chamfer_distance3]))
        minchamfer = torch.argsort(torch.Tensor([chamfer_distance1, chamfer_distance2, chamfer_distance3]))
        #minchamfer = minchamfer.cpu().detach().numpy().astype(int)
        #best_matrix = [R1, R2, R3][minchamfer]

        #                   yellow         pink          red
        #sym_plane_colors = [(1, 1, 0), (1, 0.41, 0.70), (1, 0, 0)]
        #sym_plane_colors[minchamfer] = (0, 1, 0)
        #screenshot.object_and_planar_syms(d2[0].transpose(0, 1).cpu().detach(),
        #                                  [Np1.cpu().detach().numpy(), Np2.cpu().detach().numpy(), Np3.cpu().detach().numpy()],
        #                                  [x1[0].transpose(0, 1).cpu().detach(), x2[0].transpose(0, 1).cpu().detach(), x3[0].transpose(0, 1).cpu().detach()],
        #                                  sym_plane_colors=sym_plane_colors, correct_id=minchamfer
        #                                  )

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