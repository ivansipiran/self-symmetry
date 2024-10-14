import argparse
from pathlib import Path
import os
import util
import warnings


def get_parser(name='Self-Sampling') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--export-interval', default=1000, type=int)
    parser.add_argument('--D1', default=5000, type=int)
    parser.add_argument('--D2', default=5000, type=int)
    parser.add_argument('--max-points', default=-1, type=int)
    parser.add_argument('--save-path', type=Path, required=True)
    parser.add_argument('--pc', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--stn', action='store_true')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--init-var', default=-1.0, type=float)
    parser.add_argument('--sampling-mode', default='uniform', type=str)
    parser.add_argument('--p1', default=0.9, type=float)
    parser.add_argument('--p2', default=-1.0, type=float)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--percentile', type=float, default=-1.0)
    parser.add_argument('--ang-wt', type=float, default=0.1)
    parser.add_argument('--force-normal-estimation', action='store_true')
    parser.add_argument('--kmeans', action='store_true')
    parser.add_argument('--mse', action='store_true')
    parser.add_argument('--curvature-cache', type=str, default='')
    parser.add_argument('--log-loss-save-path', type=str, default='loss_log_prueba.txt')
    parser.add_argument('--number-of-points', type=int, default=10000)

    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda2', type=float, default=1)
    parser.add_argument('--lambda3', type=float, default=1)

    parser.add_argument('--chamfer-normalize1', action='store_true')
    parser.add_argument('--chamfer-normalize2', action='store_true')

    parser.add_argument('--generator', type=Path, default='')

    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--model', type=str, default='PointNet1')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--scheduler-restart', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='oneobject', required=True)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--epochs_export_interval', type=int, default=1)


    #for DGCNN
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')


    #shapenet
    parser.add_argument('--shapenetitem', type=int, default=0,
                        help='dataset item')
    parser.add_argument('--shapenet_split', type=str, default="train")
    parser.add_argument('--shapenetsubset', type=str, default="all")
    parser.add_argument('--random_rotate', action='store_true')
    parser.add_argument('--rotate_path', type=Path, default=None)
    parser.add_argument('--groundtruth_rotation_normals_path', type=Path, default=None)
    parser.add_argument('--groundtruth_rotation_rotsym_normals_path', type=Path, default=None)
    parser.add_argument('--model_normals_path', type=Path, default=None)

    #shapenet metric eval
    parser.add_argument('--metric_eval_shapenet_model_path', type=Path, default=None)
    parser.add_argument('--metric_eval_shapenet_model_name', type=str, default=None)

    #try for rotational symmetry
    parser.add_argument('--sym_rotational', type=bool, default=False)

    #threshold 14
    parser.add_argument('--threshold', type=float, default="30")

    #screenshot
    parser.add_argument("--screenshots_enable", action='store_true')

    #resume for pointnet++ normal metric
    parser.add_argument("--resume", action='store_true')

    return parser


def parse_args(parser: argparse.ArgumentParser, inference=False):
    args = parser.parse_args()

    if args.p2 == -1.0:
        args.p2 = 1 - args.p1

    if not os.path.exists(args.save_path):
        Path.mkdir(args.save_path, exist_ok=True, parents=True)
    if not inference:
        Path.mkdir(args.save_path / 'exports', exist_ok=True, parents=True)
        Path.mkdir(args.save_path / 'targets', exist_ok=True, parents=True)
        Path.mkdir(args.save_path / 'sources', exist_ok=True, parents=True)
        Path.mkdir(args.save_path / 'generators', exist_ok=True, parents=True)

    with open(args.save_path / ('inference_args.txt' if inference else 'args.txt'), 'w+') as file:
        file.write(util.args_to_str(args))

    return args
