from torchvision import datasets, transforms
from torchsummary import summary
from torch.amp import autocast, GradScaler
import kornia.augmentation as K

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import time
import tqdm
import sys

sys.path.append("..\init\model_arch_inits")

from model_double_pool import Network as double_pool
from model_underdog import Network as underdog
from model_long_runner import Network as long_runner
from model_new_double_pool import Network as new_double_pool
from model_new_mindfuck import Network as new_mindfuck
from model_no_head import Network as no_head
from model_good_no_head import Network as good_no_head
from model_damian1 import Network as damian1 

def initialize_model(model_name):
    model_path = '../../models/' + model_name + '.pth'
    conf_mat_name = './conf_matrix/' + model_name + '_conf_matr.png'
    
    if(model_name == "new_mindfuck"):
        model = new_mindfuck()
        
    if(model_name == "new_double_pool"):
        model = new_double_pool()

    if(model_name == "long_runner"):
        model = long_runner()
        
    if(model_name == "underdog"):
        model = underdog()
        
    if(model_name == "double_pool"):
        model = double_pool()
    
    if(model_name == "no_head"):
        model = no_head()

    if(model_name == "good_no_head"):
        model = good_no_head()
        
    if(model_name == "damian1"):
        model = damian1()

    
    return model, model_path, conf_mat_name
        