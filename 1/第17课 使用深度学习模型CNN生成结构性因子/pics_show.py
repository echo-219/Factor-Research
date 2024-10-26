import  matplotlib.image as img


img_path = 'D:/9_quant_course/show.jpg'
image = img.imread(img_path)

# print(image)
# print(image.shape)
# channel

red = image[:,:,0]
green = image[:,:,1]
blue =image[:,:,2]

# print(red) # 像素点
# print(red.shape)
# print(green)
# print(green.shape)
# print(blue)
# print(blue.shape)
# print(red == blue)

import numpy as np

lags = (
    (np.arange(5) + 1).tolist()
    + (np.arange(5) + 46).tolist()
    + (np.arange(5) + (48 * 7) - 2).tolist()
)

arr = np.array([1,np.nan, 2])

arr= np.where(np.isnan(arr), 4, arr)

# print(arr)