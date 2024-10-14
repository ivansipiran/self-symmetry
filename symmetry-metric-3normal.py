from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from dgcnn.model import DGCNN_cls_orig, DGCNN_cls
from log import CDSLossLogger
from screenshot import ScreenshotsPolyscope, NullScreenshot
from shapenetdataset.shapenetdataset import ShapenetDataset
from symmetry import sel_dataloader, sel_model, sel_optimizer, sel_scheduler, unpack_elements, sel_object

try:
    import open3d
except:
    pass
import torch
import options as options
import util
import torch.optim as optim
from losses import chamfer_distance
from data_handler import get_dataset
from torch.utils.data import DataLoader
from models import SymmetryNetwork
from torch.nn import L1Loss
import sys
import os
import polyscope as ps
from pathlib import Path
import traceback
import numpy as np
import matplotlib.pyplot as plt
from Pointnet.models import pointnet2_cls_ssg

def train_oneobject_produce_metric_planesym(d2, args, screenshot=NullScreenshot()):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    print(f'device: {device}')

    logger_obj = CDSLossLogger(args)

    object = d2
    data_loader = get_dataset(args.sampling_mode)(object, device, args)

    model = sel_model(args, num_class=6)

    print(f'number of parameters: {util.n_params(model)}')
    print(f'random rotate: ', args.random_rotate)
    #model.initialize_params(args.init_var)
    model.to(device)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = sel_optimizer(model.parameters(), args)
    #scheduler = CosineAnnealingLR(optimizer, T_max = args.iterations)
    scheduler = sel_scheduler(optimizer, args)

    model.train()
    train_loader = DataLoader(data_loader, num_workers=0,
                          batch_size=args.batch_size, shuffle=False, drop_last=False)

    epoch_number = 0
    args.dataset = "oneobject"
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

            logger_obj.logloss([loss1.item(), 0, 0])
            if i % 100 == 0:
                #print(f'{args.save_path.name}; iter: {i} / {int(len(data_loader) / args.batch_size)}; Loss1: {util.show_truncated(loss1.item(), 6)}; Loss2: {util.show_truncated(loss2.item(), 6)}; Loss3: {util.show_truncated(loss3.item(), 6)}')

                logger_obj.print(i, len(data_loader), print_optim=optimizer.param_groups[0]['lr'])

        if epoch == args.epochs - 1:

            util.export_pc(output[0], args.save_path / f'targets/export_iter{epoch}_FINAL.xyz')
            # util.export_pc(d2[0], args.save_path / f'sources/export_iter:{i}.xyz')
            torch.save(model.state_dict(), args.save_path / f'generators/model{epoch}_FINAL.pt')

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

        print('Np1: ', Np1)
        print('Np2: ', Np2)
        print('Np3: ', Np3)

        R1 = util.point_and_normal_to_matrix(Np1[None, :, None], device).to(device)
        R2 = util.point_and_normal_to_matrix(Np2[None, :, None], device).to(device)
        R3 = util.point_and_normal_to_matrix(Np3[None, :, None], device).to(device)

        x = d2.transpose(2, 1)
        print(f'Input shape:{x.shape}')
        x1 = torch.bmm(x, R1)
        x2 = torch.bmm(x, R2)
        x3 = torch.bmm(x, R3)

        x1 = x1.transpose(2, 1)
        x2 = x2.transpose(2, 1)
        x3 = x3.transpose(2, 1)


        chamfer_distance1 = chamfer_distance(x1, d2, mse=args.mse)
        chamfer_distance2 = chamfer_distance(x2, d2, mse=args.mse)
        chamfer_distance3 = chamfer_distance(x3, d2, mse=args.mse)

        minchamfer = torch.argmin(torch.Tensor([chamfer_distance1, chamfer_distance2, chamfer_distance3]))
        best_matrix = [R1, R2, R3][minchamfer]

        util.export_pc(x1[0], args.save_path / 'pruebax1.xyz')
        util.export_pc(x2[0], args.save_path / 'pruebax2.xyz')
        util.export_pc(x3[0], args.save_path / 'pruebax3.xyz')

        matrixtxt = ['matriz1.txt', 'matriz2.txt', 'matriz3.txt']
        matrixtxt[minchamfer] = matrixtxt[minchamfer].split(".")[0] + "BESTMATRIX.txt"

        np.savetxt(args.save_path / matrixtxt[0], R1[0].cpu().detach().numpy())
        np.savetxt(args.save_path / matrixtxt[1], R2[0].cpu().detach().numpy())
        np.savetxt(args.save_path / matrixtxt[2], R3[0].cpu().detach().numpy())

        # #batch
        # util.export_pc(x[1], args.save_path / 'prueba1.xyz')
        # #batch

        logger_obj.write(args.log_loss_save_path) #'loss_log_prueba.txt'

        #                   yellow         pink          red
        sym_plane_colors = [(1, 1, 0), (1, 0.41, 0.70), (1, 0, 0)]
        sym_plane_colors[minchamfer] = (0, 1, 0)
        screenshot.object_and_planar_syms(d2[0].transpose(0, 1).cpu().detach(),
                                          [Np1.cpu().detach().numpy(), Np2.cpu().detach().numpy(), Np3.cpu().detach().numpy()],
                                          [x1[0].transpose(0, 1).cpu().detach(), x2[0].transpose(0, 1).cpu().detach(), x3[0].transpose(0, 1).cpu().detach()],
                                          sym_plane_colors=sym_plane_colors, correct_id=minchamfer
                                          )

        return [Np1, Np2, Np3][minchamfer]
        
def vis3(args):
    # Initialize
    ps.init()

    device = torch.device('cpu')
    #target_pc_full: torch.Tensor = util.get_input(args, center=True).unsqueeze(0).permute(0, 2, 1).to(device)

    args.pc = Path(args.save_path / "origin.xyz")
    origin = util.get_input(args, center=True).unsqueeze(0).permute(0, 2, 1).to(device)[0, :3, :].permute(1, 0)
    color_arange = np.arange(origin.shape[0])
    color = np.zeros(shape=(origin.shape[0], 3))
    color[:, 0] = color_arange/np.max(color_arange)

    ps_cloud_origin = ps.register_point_cloud("origin", origin, enabled=True)
    ps_cloud_origin.add_color_quantity("rand colors", color)

    args.pc = Path(args.save_path / "pruebax1.xyz")
    ps_cloud1 = ps.register_point_cloud("pruebax1", util.get_input(args, center=True).unsqueeze(0).permute(0, 2, 1).to(device)[0, :3, :].permute(1, 0), enabled=True)
    ps_cloud1.add_color_quantity("rand colors", color)

    args.pc = Path(args.save_path / "pruebax2.xyz")
    ps_cloud2 = ps.register_point_cloud("pruebax2", util.get_input(args, center=True).unsqueeze(0).permute(0, 2, 1).to(device)[0, :3, :].permute(1, 0), enabled=True)
    ps_cloud2.add_color_quantity("rand colors", color)

    args.pc = Path(args.save_path / "pruebax3.xyz")
    ps_cloud3 = ps.register_point_cloud("pruebax3", util.get_input(args, center=True).unsqueeze(0).permute(0, 2, 1).to(device)[0, :3, :].permute(1, 0), enabled=True)
    ps_cloud3.add_color_quantity("rand colors", color)

    # Position the camera

    # Adjust some screenshot default settings if you'd like
    ps.set_screenshot_extension(".jpg")

    prueba_filenames = ["prueba01.jpg", "prueba02.jpg", "prueba03.jpg"]

    for path in list(args.save_path.glob("matriz*.txt")):
        str_path = path.name
        if str_path.find("BESTMATRIX") != -1:
            prueba_filenames[int(str_path[6])-1] = prueba_filenames[int(str_path[6])-1].split(".")[0]+"BEST.jpg"



    # Take a screenshot
    # It will be written to your current directory as screenshot_000000.jpg, etc
    ps_cloud1.set_enabled(True)
    ps_cloud2.set_enabled(False)
    ps_cloud3.set_enabled(False)
    ps.screenshot(str(args.save_path / prueba_filenames[0]))
    ps_cloud1.set_enabled(False)
    ps_cloud2.set_enabled(True)
    ps_cloud3.set_enabled(False)
    ps.screenshot(str(args.save_path / prueba_filenames[1]))
    ps_cloud1.set_enabled(False)
    ps_cloud2.set_enabled(False)
    ps_cloud3.set_enabled(True)
    ps.screenshot(str(args.save_path / prueba_filenames[2]))

    #ps.show()


def compare_metrics(args):
    pass

def main():
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    parser = options.get_parser('Train Self-Sampling generator')
    args = options.parse_args(parser)

    root = args.metric_eval_shapenet_model_path
    d = ShapenetDataset(root=str(root), dataset_name=str(args.metric_eval_shapenet_model_name),
                        num_points=2048, split="train", random_rotate=args.random_rotate, load_name=True,
                        rotate_path=args.rotate_path, normals_path=args.groundtruth_rotation_normals_path
                        )
    if args.screenshots_enable:
        if not (args.save_path / "screenshots").exists():
            (args.save_path / "screenshots").mkdir()
        Screenshot = ScreenshotsPolyscope(args.save_path / "screenshots", enabled=args.screenshots_enable, extension="jpg")
    else:
        Screenshot = NullScreenshot()

    output_normals = []
    scrns_counter = 0

    if args.resume:
        output_normals = np.load(args.model_normals_path)
        loaded_length = output_normals.shape[0]
        output_normals = output_normals.tolist()
        print("resume")
        print(f"loaded length: {loaded_length}")

    skip_ = 0
    airplane_id = 0
    for i, (d2, label, name, file) in enumerate(d):
        print(f"%%%%%%%%%%%%%object{i}%%%%%%%%%%%%%")
        if args.screenshots_enable and scrns_counter > 20:
            break
        if name != "airplane":
            continue
        if args.resume:
            if skip_ < loaded_length:
                skip_+=1
                airplane_id+=1
                continue

        print(f"airplane {airplane_id}")
        d2 = d2.to(device)
        current_gt_normal = d.rot_normals[i].to(device)

        Screenshot.object_id = f"object{i}"

        args.shapenetitem = i
        args.dataset = "shapenet"
        train_output = train_oneobject_produce_metric_planesym(d2, args, Screenshot)
        output_normals.append(train_output.cpu().detach().numpy())
        print(f"output normal: {train_output}")
        print(f"GT normal: {current_gt_normal}")
        print(f"comparison to ground truth (1 is better): {torch.abs(1 - torch.abs(torch.nn.functional.cosine_similarity(train_output, current_gt_normal, dim=0)))}")

        np.save( args.model_normals_path, np.array(output_normals))
        np.savetxt(str(args.model_normals_path) + ".txt", np.array(output_normals))
        
        scrns_counter+=1
        airplane_id+=1
    #vis3(args)
    #plot(args)
    compare_metrics(args)
    print("done")


def mainw7():
    try:
        main()
    except Exception as e:
        traceback.print_exc()
    finally:
        pid = os.getpid()
        # print(pid)
        os.system(f"taskkill /f /pid {pid}")

if __name__ == "__main__":
    mainw7()