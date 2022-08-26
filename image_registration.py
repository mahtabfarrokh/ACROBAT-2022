from skimage.feature import canny
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
# import tifffile
import rasterio as rio
import data_aug.stain_utils as stain_utils
import data_aug.stainNorm_Macenko as stainNorm_Macenko

Image.MAX_IMAGE_PIXELS = 1000000000

def apply_canny(path, slide_name, normalizer, norm= True):
    with rio.open(path + slide_name) as img:
        imgnp = img.read()
        imgnp = np.array(imgnp)
        img = np.transpose(imgnp, (1, 2, 0))
        img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
        if norm:
            img = normalizer.transform(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img


    edges = canny(img / 255., sigma=1.4)
    print(edges.shape)
    # fig, ax = plt.subplots(figsize=(4, 3))
    # ax.imshow(edges, cmap=plt.cm.gray)
    # ax.axis('off')
    # ax.set_title('Canny detector')
    # plt.show()
    edges = edges.astype(np.uint8)
    edges *= 255
    return edges

def generate_points(ihc_x, ihc_y, H):
    p1 = [ihc_x*0.1, ihc_y*0.1, 1]
    p2 = H @ p1
    return (p2[0]*10.0000000000)/ p2[2] , (p2[1]*10.0000000000)/ p2[2]


path = "/Users/mahtabfarrokh/PycharmProjects/pythonProject/ACROBAT/dataset/"
csv_file = "acrobat_test_points_public_1_of_1.csv"
# image_dir = "acrobat_validation_pyramid_1_of_1/"
image_dir = "mnt/hdd8tb/acrobat_wsis_test_anon_pyramidal_tiff/"
outout = "acrobat_test_points_public_1_of_1_out.csv"
res_he_x = []
res_he_y = []
images_dataset = pd.read_csv(path + csv_file)
anon_id = -1
H = 0
print("Here..")
i1 = stain_utils.read_image("./acrobat_train_x5_example/100_HE_x5_z0.tif")
normalizer = stainNorm_Macenko.Normalizer()
normalizer.fit(i1)
print("Done fitting..")
for i in range(len(images_dataset)):
    if i % 10 == 0:
        print(i)
    if anon_id == images_dataset["anon_id"][i]:
        x, y = generate_points(images_dataset["ihc_x"][i], images_dataset["ihc_y"][i], H)
        res_he_x.append(x)
        res_he_y.append(y)
        continue

    anon_id = images_dataset["anon_id"][i]
    slide = images_dataset["anon_filename_ihc"][i].split(".")[0] + ".tiff"
    first_img = apply_canny(path + image_dir, slide, normalizer, True)
    slide = images_dataset["anon_filename_he"][i].split(".")[0] + ".tiff"
    second_img = apply_canny(path + image_dir, slide, normalizer, False)
    height, width = second_img.shape

    orb_detector = cv.ORB_create(100000)
    kp1, d1 = orb_detector.detectAndCompute(first_img, None)
    kp2, d2 = orb_detector.detectAndCompute(second_img, None)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches = list(matches)
    matches.sort(key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.3)]
    no_of_matches = len(matches)
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for j in range(len(matches)):
        p1[j, :] = kp1[matches[j].queryIdx].pt
        p2[j, :] = kp2[matches[j].trainIdx].pt

    homography, mask = cv.findHomography(p1, p2, cv.RANSAC)
    H = homography
    x, y = generate_points(images_dataset["ihc_x"][i], images_dataset["ihc_y"][i], H)
    res_he_x.append(x)
    res_he_y.append(y)

    if i < 200:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(first_img, cmap=plt.cm.gray)
        ax.plot(images_dataset["ihc_x"][i]/10.0, images_dataset["ihc_y"][i]/10.0, 'o', color='yellow')
        ax.axis('off')
        ax.set_title('points'+ slide)
        plt.show()

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(second_img, cmap=plt.cm.gray)
        ax.plot(x/10.0, y/10.0, 'o', color='yellow')
        ax.axis('off')
        ax.set_title('points' + slide)
        plt.show()

    if i%2 == 0:
        dic_output = {}
        dic_output["he_x"] = res_he_x
        dic_output["he_y"] = res_he_y
        print(dic_output)
        df = pd.DataFrame.from_dict(dic_output)
        df.to_csv(outout, index=False)
        print("Wrote...")


dic_output = {}
dic_output["he_x"] = res_he_x
dic_output["he_y"] = res_he_y
df = pd.DataFrame.from_dict(dic_output)
df.to_csv(outout,  index=False)

# slide = "99_ER_x5_z0.tif"
# first_img = apply_canny(path, slide)
# slide = "99_HE_x5_z0.tif"
# second_img = apply_canny(path, slide)
# print(second_img.shape)
# height, width = second_img.shape
# # first_img = cv.resize(first_img, (width, height) , interpolation = cv.INTER_AREA)
#
#
# orb_detector = cv.ORB_create(50000)
# kp1, d1 = orb_detector.detectAndCompute(first_img, None)
# kp2, d2 = orb_detector.detectAndCompute(second_img, None)
# matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
# matches = matcher.match(d1, d2)
# matches = list(matches)
# print(len(matches))
# matches.sort(key=lambda x: x.distance)
# matches = matches[:int(len(matches) * 0.2)]
# no_of_matches = len(matches)
# print(no_of_matches)
# p1 = np.zeros((no_of_matches, 2))
# p2 = np.zeros((no_of_matches, 2))
#
# for i in range(len(matches)):
#     p1[i, :] = kp1[matches[i].queryIdx].pt
#     p2[i, :] = kp2[matches[i].trainIdx].pt
#
#
# homography, mask = cv.findHomography(p1, p2, cv.RANSAC)
# transformed_img = cv.warpPerspective(first_img,
#                                       homography, (width, height))
#
#
# cv .imwrite('output.jpg', transformed_img)
