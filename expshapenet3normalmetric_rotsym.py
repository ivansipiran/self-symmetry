import os
from pathlib import Path

from shapenetdataset.shapenetdataset import ShapenetDataset

modelpath = "/data/gmontana/"
modellist = []

day = "4-29-2"
savepath = Path(f"/data/gmontana/results/{day}")
commentpath = savepath / "comment.txt"
comment = "DGCNN, Calculate cosine distance on Shapenet bottle subset using 3 normals centered (0,0,0)," \
          " iters per item 500, Adam, batches 1, epochs 1, CosineAnnealingLR, " \
          "chamfer loss and force normals to be orthonormal loss, no chamfer normalize"

if not savepath.exists():
    savepath.mkdir()
with open(commentpath, "w+") as f:
    f.write(comment)

lrlist = [0.01]

lambdalist1 = [1]
lambdalist2 = [1]
lambdalist3 = [1]

nets = ["Dgcnn_3vector"]

mpath = "shapenetcorev2"
modelname = "shapenetcorev2"
rotate_path = str(Path(f"/data/gmontana/results/{day}/random_rotations_matrix.npy"))
rot_normals_path = str(Path(f"/data/gmontana/results/{day}/rot_normals_rotsym.npy"))
model_normals_path = str(Path(f"/data/gmontana/results/{day}/model_normals.npy"))


for net in nets:
    for lr in range(len(lrlist)):
        for i in range(len(lambdalist1)):
            os.system(f"python symmetry-metric-3normalrot.py --lr {lrlist[lr]} --name {mpath} --iterations 500 --export-interval 100 --pc "
                  f" {Path(modelpath) / Path(mpath)} --init-var 0.15 --D1 8000 --D2 8000 --sampling-mode curvature "
                  f"--save-path {Path(str(savepath)+'/'+mpath+'-'+f'{net}'+f'bottles-cosinesimilarity-'+'-'+str(lrlist[lr])+'+'+str(lambdalist1[i])+'+'+str(lambdalist2[i])+'+'+str(lambdalist3[i]))} "
                  f"--batch-size 1 --k 40 --p1 0.85 --p2 0.2 --force-normal-estimation --mse "
                  f"--optimizer Adam --model {net} --scheduler CosineLR "
                  f"--epochs 1 "
                  f"--epochs_export_interval 10 "
                  f"--random_rotate "
                  f"--rotate_path {rotate_path} "
                  f"--groundtruth_rotation_rotsym_normals_path {rot_normals_path} "
                  f"--model_normals_path {model_normals_path} "
                  f"--dataset shapenet "
                  f"--sym_rotational True "
                  f"--metric_eval_shapenet_model_path {Path(modelpath)/Path(mpath)} "
                  f"--metric_eval_shapenet_model_name {modelname} "
                  f"--lambda1 {lambdalist1[i]} "
                  f"--lambda2 {lambdalist2[i]} "
                  f"--lambda3 {lambdalist3[i]} ")