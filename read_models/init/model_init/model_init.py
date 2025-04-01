import sys

sys.path.append("./read_models/init/model_arch_inits")

from model_damian1 import Network as damian1 
from model_hubert1 import Network as hubert1
from model_hubert2 import Network as hubert2
from model_uberdriver79 import Network as uberdriver79

def initialize_model(model_name):
    model_path = './models/' + model_name + '.pth'
    conf_mat_name = './read_models/model_analysis/conf_matrix/' + model_name + '_conf_matr.png'
    
    if("damian1" in model_name):
        model = damian1()

    if("hubert1" in model_name):
        model = hubert1()
    
    if("hubert2" in model_name):
        model = hubert2()

    if("uberdriver79" in model_name):
        model = uberdriver79()
    
    return model, model_path, conf_mat_name