import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def denormalize(T,coords):
    return 0.5 * ((coords + 1.0)*T)

def img2array(data_path,desired_size = None,expand = False,view = False):
    img = Image.open(data_path)
    img = img.convert("RGB")
    if desired_size:
        img = img.resize(desired_size[1],desired_size[0])
    if view:
        img.show()
    x = np.array(img,type = "float32")
    if expand :
        x = np.expand_dims(x,axis = 0)
    x/=255.0
    return x

def array2img(x):
    x = np.array(x)
    x = x+max(-np.min(x),0)#マイナス値があった時に最小の値を0にする
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x*= 255
    return Image.fromarray(x.astype("uint8"),"RGB")