import faulthandler
import torch
from utils import parser
from utils.config import *
from datasets.ShapeNetDataset import *
from data_handler import get_dataset
from symmetry import sel_dataloader, sel_model, sel_optimizer, sel_scheduler

faulthandler.enable()

if __name__=="__main__":
    print('aja')
    