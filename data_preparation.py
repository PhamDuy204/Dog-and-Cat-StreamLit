import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from preprocessing import *
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
cat_image=glob.glob('./Data/data/Cat/*.jpg')
dog_image=glob.glob('./Data/data/Dog/*.jpg')
def get_data():
    with ThreadPoolExecutor(max_workers=4) as executor:
        cat_feature=list(executor.map(
            lambda x:preprocessing(x)[0],cat_image
        ))
        labels = len(cat_feature)*[0]
        dog_feature=list(executor.map(
            lambda x:preprocessing(x)[0],dog_image
        ))
        labels+=len(dog_feature)*[1]
    return  np.array(cat_feature+dog_feature),np.array(labels)


