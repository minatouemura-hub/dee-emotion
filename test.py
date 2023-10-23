from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
import torch
from retina import Retina
from functools import reduce
from utils import array2img
from PIL import Image

#画像の読み込み
path = "/Users/uemuraminato/Desktop/deep_emotion/92659b85-9b4b-455c-9952-e63884e898fd.jpeg"
x = read_image(path)
img = x.unsqueeze(0)
print(img.shape)

#パラメータの設定
loc = torch.from_numpy(np.array([[0.0,0.0]]))#l
patch_size = 50 #g
num_patches = 3#k
scale = 2    #s

ret  = Retina(g = patch_size,k = num_patches,s = scale)

#test_extracted
extracted = ret.extract_patch(img,loc,200)
print(extracted.shape)
extracted = extracted.squeeze(0).detach().numpy()
print(extracted.shape)

def merge_images(image1, image2):
    """Merge two images into one, displayed side by side.
    """
    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new("RGB", (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result

#test_foveate
glimpse = ret.foveate(img,loc).detach().numpy()
glimpse = np.reshape(glimpse,[1,num_patches,3,patch_size,patch_size])
glimpse = np.transpose(glimpse,[0,1,3,4,2])
merged = []

for i in range(len(glimpse)):
    g = glimpse[i]
    g = list(g)
    g = [array2img(l) for l in g]
    res = reduce(merge_images,list(g))
    merged.append(res)

merged = [np.asarray(l,dtype="float32")/255.0 for l in merged]

fig,ax = plt.subplots()
for i in range(len(merged)):
    ax.imshow(merged[i])
ax.set_xticks([])
ax.set_yticks([])
plt.show()
