import sys

sys.path.append("./read_models/init/model_arch_inits")

from model_good_no_head import Network as good_no_head
from model_damian1 import Network as damian1 
from model_hubert1 import Network as hubert1
from model_hubert2 import Network as hubert2
from model_3_head import Network as head_3
from model_2_head import Network as head_2
from model_1_head import Network as head_1
from model_five_twelve import Network as five_twelve

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