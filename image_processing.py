import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob
from imgaug import augmenters as iaa
from PIL import Image
from tqdm.notebook import tqdm

image_list = os.listdir('C:/Users/Research/anaconda3/Visual_Python3/OSA_PSG/new_image')
print(image_list[:5])
print(len(image_list))

glob_image = glob('C:/Users/Research/anaconda3/Visual_Python3/OSA_PSG/new_image/*.jpg')
print('How many glob list?:', len(glob_image))

df = pd.read_csv('df2_merge_toy.csv')
print(df)

# Unbalancing image & csv list(number). So, We will equalize to them.
# First, Matching image processing
'''
img = cv2.imread('raw_images_merge/10001149_20200824_20150909142106_Cephalometry.jpg')
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf, 0)

# History Equalization
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

# Mask processing = exchange to 0(zero)
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[img]
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(img2), plt.title('Equalization')
plt.show()

This code is Equalization to numpy
But, I mean.. it is  just Equalization.
So, we will find to Image Characteristics
'''

# # and..
# img = cv2.imread('raw_images_merge/10001149_20200824_20150909142106_Cephalometry.jpg', 0)
#
# # contrast limit is 2. title size is 8x8
# xray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img2 = xray.apply(img)
#
# img = cv2.resize(img, (600, 600))
# img2 = cv2.resize(img2, (600, 600))
#
# dst = np.hstack((img, img2))
# # print(dst)
# cv2.imshow("X_ray image", dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

'''
Finish! and Make function
input is raw_iamges_merge -> output is new_processing_image
ex)
>>> change_img('C:/Users/Research/anaconda3/Visual_Python3/OSA_PSG/raw_images_merge')
stack to new_processing_img(be changed)
'''

# change_image = []
# path = 'C:/Users/Research/anaconda3/Visual_Python3/OSA_PSG/new_processing_image'
# for i in glob_image:
#     i = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     change_image.append(i)
#

#
# print(len(change_image))
# print(change_image[:5])
#
# new_df = pd.DataFrame(change_image)
# print(new_df)

# MemoryError가 발생했으므로, 이미지 크기를 줄여본다.
for f in tqdm(glob_image):
    img = Image.open(f)
    img_resize = img.resize((700, 700))
    title, ext = os.path.splitext(f)
    img_resize.save(title + ext)

# 이미지 읽어오는 함수
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# 이미지 저장 함수
def write_images(images):
    for i in range(0, len(images)):
        cv2.imwrite('C:/Users/Research/anaconda3/Visual_Python3/OSA_PSG/data_augmentation', images[i])
    print("image saving complete")

# 여러 폴더에 한번에 저장하기
def imagewriterfunction(folder, images):
    for i in range(0, len(images)):
        write_images(folder, str(i), images[i])
    print("all images saved to folder")

# 이미지 증강 코드
def augmentations1(images):
    seq1 = iaa.Sequential([
        iaa.AverageBlur(k=(2, 7)),
        iaa.MedianBlur(k=(3, 11))
    ])

    seq2 = iaa.ChannelShuffle(p=1.0)
    seq3 = iaa.Dropout((0.05, 0.1), per_channel=0.5)
    seq4 = iaa.Sequential([
        iaa.Add((-15, 15)),
        iaa.Multiply((0.3, 1.5))
    ])
    print("image augmentation beginning")
    img1 = seq1.augment_images(images)
    print("sequence 1 completed......")
    img2 = seq2.augment_images(images)
    print("sequence 2 completed......")
    img3 = seq3.augment_images(images)
    print("sequence 3 completed......")
    img4 = seq4.augment_images(images)
    print("sequence 4 completed......")
    print("proceed to next augmentations")
    list = [img1, img2, img3, img4]
    return list


folder = 'C:/Users/Research/anaconda3/Visual_Python3/OSA_PSG/new_image'
photos = os.listdir(folder)

photos1 = load_images_from_folder(folder)
photo_augmented1234 = augmentations1(photos1)
write_images(photo_augmented1234)