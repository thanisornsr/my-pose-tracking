import cv2
import glob
import os

input_images_dir = './live_imgs'
FPS = 3
for filename in sorted(glob.glob('./live_output.mp4')):
    os.remove(filename)

img_array = []
for filename in sorted(glob.glob(input_images_dir+'/*.jpg')):
    img = cv2.imread(filename)
    i_h, i_w,_ = img.shape
    img_size = (i_w,i_h)
    img_array.append(img)

four_cc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('live_output.mp4',four_cc, FPS, img_size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
