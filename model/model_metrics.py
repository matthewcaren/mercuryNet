# from inference import load_model, infer_vid
import torch
from shutil import copy
from torchvision import transforms
from PIL import Image
from pathlib import Path
from glob import glob
import numpy as np
from tqdm import tqdm
import sys, cv2, os, pickle, argparse, subprocess

from model import Decoder, Encoder
from hparams import hparams as hps
from utils.util import mode, to_var, to_arr

model = Encoder()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("total trainable weights:", params)
