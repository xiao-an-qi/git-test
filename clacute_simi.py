import os
import random

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
# import matplotlib.pyplot as plt


# 灰度共生矩阵 (GLCM)
def compute_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    features = [contrast, dissimilarity, homogeneity, energy, correlation, asm]
    # features = [contrast, energy, correlation, asm]
    return np.array(features)


def compute_glcm_features_color(image):
    glcm_features = []
    for channel in cv2.split(image):
        glcm = graycomatrix(channel, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        glcm_features.extend([contrast, dissimilarity, homogeneity, energy, correlation, asm])
    return np.array(glcm_features)


# 局部二值模式 (LBP)
def compute_lbp_features(image):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# 梯度直方图 (HOG)
def compute_hog_features(image):
    hog_features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_features, hog_image


def cal_simi(vector1, vector2, cos_flag=True):
    # 计算欧氏距离
    if cos_flag:
        simi = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    else:
        simi = np.linalg.norm(vector1 - vector2)

    return simi


# 读取图像
target_width = 300
target_height = 300

# image = cv2.imread(r'C:\Users\79258\Desktop\UnderKill0\2EX0AZ6M1U0/387362291#256.28-16.51-21.6@2EX0AZ6M1U0.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread(r'C:\Users\79258\Desktop\UnderKill0\2EX0AZ6M1U0/387362291#256.28-16.51-21.6@2EX0AZ6M1U0.jpg')
image = cv2.resize(image, (target_width, target_height))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
glcm_features = compute_glcm_features_color(image)
# glcm_features = compute_lbp_features(image)
# print(glcm_features)

devices_dir = r'C:\Users\79258\Desktop\UnderKill0/'
devices = os.listdir(devices_dir)
# devices = ['2EX0EQGP1U1']
for device in devices:
    try:
        device_dir = devices_dir + device
        images = os.listdir(device_dir)
        random.shuffle(images)
        image_address = os.path.join(device_dir, images[0])
        image1 = cv2.imread(image_address)
        image1 = cv2.resize(image1, (target_width, target_height))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        # glcm_features1 = compute_glcm_features(image1)
        glcm_features1 = compute_glcm_features_color(image1)

        # print(glcm_features1)
        cos_sim = cal_simi(glcm_features, glcm_features1, False)
        print(device, cos_sim)
    except:
        continue

# image2 = cv2.imread(r'C:\Users\79258\Desktop\UnderKill0\2EX0BHEM2U0/391314301#21550.86-579.87-412.16@2EX0BHEM2U0.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread(r'C:\Users\79258\Desktop\UnderKill0\2EX0BYTP2U1/389749126#243.69-18.45-28.38@2EX0BYTP2U1.jpg', cv2.IMREAD_GRAYSCALE)
# glcm_features2 = compute_glcm_features(image2)
# print(glcm_features2)
#
# # 计算余弦相似度
# cos_sim = cal_simi(glcm_features, glcm_features1, False)
# print(f'cos_sim: {cos_sim}.')
#
# cos_sim = cal_simi(glcm_features, glcm_features2, False)
# print(f'cos_sim: {cos_sim}.')

# max_sim_value = np.max(cos_sim)
# max_sim_index = np.argmax(cos_sim)


# 计算 LBP 特征
# lbp_features = compute_lbp_features(image)
# print(f"LBP Features: {lbp_features}")
# print(f"LBP Features len: {len(lbp_features)}")
#
# # 计算 HOG 特征并显示 HOG 图像
# hog_features, hog_image = compute_hog_features(image)
# print(f"HOG Features: {hog_features}")
# print(f"HOG Features len: {len(hog_features)}")

# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.title('HOG Image')
# plt.imshow(hog_image, cmap='gray')
# plt.show()
