import os, json, cv2, random, math, glob

images_dir = './temp'
img_array = []
for filename in sorted(glob.glob(images_dir+'/*.jpg')):
    img = cv2.imread(filename)
    i_h, i_w,_ = img.shape
    img_size = (i_w,i_h)
    img_array.append(img)

four_cc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('OT_single_vid0_output.mp4',four_cc, 8, img_size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()