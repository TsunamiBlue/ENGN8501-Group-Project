"""
Deep learning, testing the entire steps.
Should feed the same person as the target person.
"""
import os
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
import warnings

warnings.filterwarnings('ignore')
mainpath = os.getcwd()
data_dir = Path(mainpath+'/data/')
sys.path.append(str(data_dir))
pix2pixhd_dir = Path('./src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import src.config.test_opt as opt

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)


pth = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

model = create_model(opt)

for data in tqdm(dataset):
    minibatch = 1
    generated,_ = model.inference(data['label'])

    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    # print(img_path)
    visualizer.save_images(pth, visuals, img_path)

torch.cuda.empty_cache()