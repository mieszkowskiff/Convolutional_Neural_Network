from small_damian1_hubert1_hubert2_HEAD import MetaStackingHead as small
from small2_good_no_head_damian1_hubert1_hubert2_HEAD import MetaStackingHead as small2

def initialize_head(head_name):
    
    if(head_name == 'small_damian1_hubert1_hubert2_HEAD'):
        head = small()

    if(head_name == 'small2_good_no_head_damian1_hubert1_hubert2_HEAD'):
        head = small2()

    return head