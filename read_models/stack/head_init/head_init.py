
from small_head import MetaStackingHead as small

def initialize_head(head_name):
    
    if("small" in head_name):
        head = small()
    
    return head