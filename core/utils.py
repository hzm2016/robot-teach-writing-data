import numpy as np
from scipy.spatial.distance import cdist
import munkres


def load_class(class_name):
    """[summary]

    Args:
        class_name ([str): name of the class to be built

    Returns:
        [type]: [description]
    """
    components = class_name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def squarify(M,val=10000):
    (a,b)=M.shape

    if a == b:
        return M

    if a>b: 
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))

    return np.pad(M,padding,mode='constant',constant_values=val)


def pad_length(a, b, val=1000):

    len_a = len(a)
    len_b = len(b)

    if len_a == len_b:        
        return a, b, len_a, 0
    
    if len_a > len_b: 
        padding=((0,len_a - len_b),(0,0))
        b = np.pad(b,padding,mode='constant',constant_values=val)
    else:
        padding=((0,len_b - len_a),(0,0))
        a = np.pad(a,padding,mode='constant',constant_values=val)
        
    return a, b, min(len_a, len_b), int(len_a > len_b)

def hungarian_matching(a, b):
    m = munkres.Munkres()
    a, b, min_length, index  = pad_length(a, b)
    dist = cdist(a,b)
    matching = m.compute(dist)
    
    if index == 0:
        matching = matching[:min_length]
    else:
        matching = sorted(matching, key=lambda x: x[1], reverse=False)
        matching = matching[:min_length]

    return matching

if __name__ == "__main__":

    b = np.array([(1, 2),(3,4),(5,6)])
    a = np.array([(3, 4),(7,8)])

    indices = hungarian_matching(a, b)
    print(indices)
