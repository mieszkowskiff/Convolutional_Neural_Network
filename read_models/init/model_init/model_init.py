import sys

sys.path.append("..\init\model_arch_inits")

from model_good_no_head import Network as good_no_head
from model_damian1 import Network as damian1 
from model_hubert1 import Network as hubert1
from model_hubert2 import Network as hubert2
from model_3_head import Network as head_3
from model_2_head import Network as head_2
from model_1_head import Network as head_1
from model_five_twelve import Network as five_twelve

def initialize_model(model_name, tuned = False):
    if(not tuned):
        model_path = '../../models/' + model_name + '.pth'
    else:
        model_path = '../further_train/fine_tuned_models/' + model_name + '_TUNED.pth'

    conf_mat_name = './conf_matrix/' + model_name + '_conf_matr.png'

    if(model_name == "good_no_head"):
        model = good_no_head()
        
    if(model_name == "damian1"):
        model = damian1()

    if(model_name == "3_head"):
        model = head_3()
    
    if(model_name == "2_head"):
        model = head_2()

    if(model_name == "1_head"):
        model = head_1()
    
    if(model_name == "hubert1"):
        model = hubert1()

    if(model_name == "hubert2"):
        model = hubert2()
        
    if(model_name == "five_twelve"):
        model = five_twelve()

    return model, model_path, conf_mat_name