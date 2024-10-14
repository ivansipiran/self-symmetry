import torch
import numpy as np
import math
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import open3d
#from torch_cluster import knn_graph



def load_stl(args):
    file_path = args.pc
    mesh = open3d.cpu.pybind.io.read_triangle_mesh(str(file_path))
    #points = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    points_sample = torch.tensor(np.asarray(mesh.sample_points_uniformly(args.number_of_points).points), dtype=torch.float32)
    #points_sample = torch.tensor(sample(mesh), dtype=torch.float32)
    print("estimating normals (stl)")
    points_6 = estimate_normals_torch(points_sample, args.k)

    return points_6

"""
def density(pc: torch.Tensor, k=10):
    knn = knn_graph(pc[:, :3], k, loop=False)
    knn_indx, _ = knn.view(2, pc.shape[0], k)
    knn_data = pc[knn_indx, :]
    max_distance, _ = (knn_data[:, :, :3] - pc[:, None, :3]).norm(dim=-1).max(dim=-1)
    dense = k / (max_distance ** 3)
    inf_mask = torch.isinf(dense)
    max_val = dense[~inf_mask].max()
    dense[inf_mask] = max_val
    return dense
"""

def get_input(args, center=False) -> torch.Tensor:
    if args.pc is not None:
        if args.pc.suffix == ".xyz":
            with open(args.pc) as file:
                pc = xyz2tensor(file.read(), force_normals=args.force_normal_estimation, k=args.k)
        elif args.pc.suffix == ".stl":
            pc = load_stl(args)
    else:
        raise NotImplementedError('no recognized input type was found')

    if center:
        cm = pc[:, :3].mean(dim=0)
        pc[:, :3] = pc[:, :3] - cm

    return pc


def truncate(number: float, digits: int) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def show_truncated(n: float, digits: int):
    x = str(truncate(n, digits))
    if len(x) < digits + 2:
        x += '0' * (digits + 2 - len(x))
    return x


def n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def xyz2tensor(txt, append_normals=False, estimate_normals_flag=True, force_normals=False, k=20):
    pts = []
    for line in txt.split('\n'):
        line = line.strip()
        spt = line.split(' ')
        if 'nan' in line:
            continue
        if len(spt) == 6:
            pts.append(torch.tensor([float(x) for x in spt]))
        if len(spt) == 3:
            t = [float(x) for x in spt]
            if append_normals and not estimate_normals_flag:
                t += [0.0 for _ in range(3)]
            pts.append(torch.tensor(t))

    rtn = torch.stack(pts, dim=0)
    if (rtn.shape[1] == 3 and estimate_normals_flag) or force_normals:
        print('estimating normals')
        rtn = estimate_normals_torch(rtn, k)
    return rtn


def angle_diff(pc, k):
    INNER_PRODUCT_THRESHOLD = math.pi / 2
    knn = knn_graph(pc[:, :3], k, loop=False)
    knn, _ = knn.view(2, pc.shape[0], k)

    inner_prod = (pc[knn, 3:] * pc[:, None, 3:]).sum(dim=-1)
    inner_prod[inner_prod > 1] = 1
    inner_prod[inner_prod < -1] = -1
    angle = torch.acos(inner_prod)
    angle[angle > INNER_PRODUCT_THRESHOLD] = math.pi - angle[angle > INNER_PRODUCT_THRESHOLD]
    angle = angle.sum(dim=-1)
    return angle


def scalar_to_color(scalar_tensor, minn=-1, maxx=1):
    jet = plt.get_cmap('jet')
    if minn == -1 or maxx == -1:
        cNorm = colors.Normalize(vmin=scalar_tensor.min(), vmax=scalar_tensor.max())
    else:
        cNorm = colors.Normalize(vmin=minn, vmax=maxx)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    p = scalarMap.to_rgba(scalar_tensor.cpu())
    return torch.tensor(p).to(scalar_tensor.device).permute(1, 0)[:3, :]


def estimate_normals_torch(inputpc, max_nn):
    knn = knn_graph(inputpc[:, :3], max_nn, loop=False)
    try:
        knn = knn.view(2, inputpc.shape[0], max_nn)[0]
    except RuntimeError as e:
        if knn.shape[0] * knn.shape[1] - 2*inputpc.shape[0]*max_nn != 0:
            knn = knn_graph(inputpc[:, :3], max_nn, loop=True)
            knn = knn.view(2, inputpc.shape[0], max_nn)[0]
        else:
            raise RuntimeError
    x = inputpc[knn][:, :, :3]
    temp = x[:, :, :3] - x.mean(dim=1)[:, None, :3]
    cov = temp.transpose(1, 2) @ temp / x.shape[0]
    #e, v = torch.symeig(cov, eigenvectors=True) ##DEPRECATED https://docs-preview.pytorch.org/73676/generated/torch.symeig.html
    e, v = torch.linalg.eigh(cov, UPLO="U")
    n = v[:, :, 0]
    return torch.cat([inputpc[:, :3], n], dim=-1)


def args_to_str(args):
    d = args.__dict__
    txt = ''
    for k in d.keys():
        txt += f'{k}: {d[k]}\n'
    return txt.strip('\n')


def voxel_downsample(point_cloud: torch.Tensor, size=0.003, npoints=-1, max_iters=int(1e2)):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    normals = point_cloud.shape[1] == 6
    if normals:
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:])
    if npoints == -1:
        pcd = pcd.voxel_down_sample(size)
        return torch.tensor(pcd.points)

    upper = 0.0001
    lower = 0.5
    for i in range(max_iters):
        mid = (upper + lower) / 2
        tmp = pcd.voxel_down_sample(mid)

        # minimal grid quantization, maximal resolution
        if np.asanyarray(tmp.points).shape[0] > npoints:
            upper = mid
        else:
            lower = mid
    if normals:
        pts = torch.tensor(tmp.points).to(point_cloud.device).type(point_cloud.dtype)
        n = torch.tensor(tmp.normals).to(point_cloud.device).type(point_cloud.dtype)
        return torch.cat([pts, n], dim=-1)
    else:
        return torch.tensor(tmp.points).to(point_cloud.device).type(point_cloud.dtype)


def export_pc(pc, dest, color=None):
    txt = ''

    def t2txt(t):
        return ' '.join(map(lambda x: str(x.item()), t))

    if color is None:
        for i in range(pc.shape[1]):
            txt += f'{t2txt(pc[:, i])}\n'
        txt.strip()
    else:
        for i in range(pc.shape[1]):
            txt += f'v {t2txt(pc[:3, i])} {t2txt(color[:3, i])}\n'
        txt.strip()
        dest = str(dest).replace('.xyz', '.obj')

    with open(dest, 'w+') as file:
        file.write(txt)


class NoScheduler():
    def __init__(self):
        pass

    def step(self):
        pass


def point_and_normal_to_matrix(n, device):
    """

    :param n: batches x 3 x 1
    """

    ident = torch.eye(3)[None, :, :].to(device)
    t_p = n.transpose(1, 2)
    p__t_p_product = 2 * n @ t_p
    return ident - p__t_p_product

def transform_object(object_points, plane_normal, point_in_plane, device):
    """

    :param object_points: batches x dims x n_points
    :param plane_normal: batches x dims x 1
    :param point_in_plane: batches x dims x 1
    :return: batches x dims x n_points
    """
    plane_normal_norm = torch.linalg.vector_norm(plane_normal, dim=1)
    plane_normal_unit = plane_normal / (plane_normal_norm[:, :, None])
    reflection_matrix = point_and_normal_to_matrix(plane_normal_unit, device)
    return reflection_matrix @ (object_points - point_in_plane) + point_in_plane, reflection_matrix



def  rotate_pointcloud_usingnormal_1_15(pointcloud, normal, angle, device):
    """


    :param pointcloud: batches x n_points x dims
    :param normal: batches x dims x 1
    :param angle: angle in radians, math.pi/2 for 90 degrees
    :return: batches x n_points x dims
    """
    batches = pointcloud.shape[0]
    ident = torch.eye(3)[None, :, :].to(device) # 1 x 3 x 3
    omega = torch.zeros(batches, 3, 3).to(device)  # 2 x 3 x 3-
    omega[:, 1, 0] = normal[:, 2, 0]
    omega[:, 2, 0] = -normal[:, 1, 0]
    omega[:, 0, 1] = -normal[:, 2, 0]
    omega[:, 2, 1] = normal[:, 0, 0]
    omega[:, 0, 2] = normal[:, 1, 0]
    omega[:, 1, 2] = -normal[:, 0, 0]

    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)

    #omega2 = torch.bmm(omega, omega)
    omega2 = omega @ omega

    R = ident + omega*sin_angle + omega2*(1 - cos_angle) # 2 x 3 x 3
    return pointcloud @ R


def get_sym_matrix(Np):
    R = torch.zeros((Np.shape[0], 3, 3))
    R[:,0,0] = 1 - 2*Np[:,0]*Np[:,0]
    R[:,0,1] = -2*Np[:,0]*Np[:,1]
    R[:,0,2] = -2*Np[:,0]*Np[:,2]
    R[:,1,0] = -2*Np[:,1]*Np[:,0]
    R[:,1,1] = 1 - 2*Np[:,1]*Np[:,1]
    R[:,1,2] = -2*Np[:,1]*Np[:,2]
    R[:,2,0] = -2*Np[:,2]*Np[:,0]
    R[:,2,1] = -2*Np[:,2]*Np[:,1]
    R[:,2,2] = 1 - 2*Np[:,2]*Np[:,2]
    return R

def get_rot_matrix(Np, angle):
    angle = angle * np.pi/180

    S = torch.zeros((Np.shape[0], 3, 3))
    S[:, 0, 1] = -Np[:, 2]
    S[:, 0, 2] = Np[:, 1]
    S[:, 1, 0] = Np[:, 2]
    S[:, 1, 2] = -Np[:, 0]
    S[:, 2, 0] = -Np[:, 1]
    S[:, 2, 1] = Np[:, 0]

    R = torch.eye(S.size(-1), dtype=S.dtype, device=S.device) + torch.mul(torch.sin(torch.tensor(angle)),S) + torch.mul(1 - torch.cos(torch.tensor(angle)),torch.bmm(S,S))

    return R


