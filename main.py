import os
import numpy as np
import torch
import torchvision.transforms as T
import models
from torch.utils.data import dataloader,DataLoader
from tqdm import tqdm


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    ### data pre-processing





    ### hyper-parameters



    ### training






if __name__ == "__main__":
    main()
