import os
import cv2

root_folder = 'VID02-training'
# (353, 215), (353, 265), (377, 265), (377, 215)
with open(os.path.join(root_folder, 'labels', '000144.txt'), 'r') as ifile:
    labels = ifile.readlines()
im = cv2.imread(os.path.join(root_folder, 'JPEGImages', '000144.jpg'))
shape_im = im.shape
print(shape_im)
for line in labels:
    line = line.strip()
    line = line.split(' ')
    line = [float(i) for i in line]
    xc = line[1] * shape_im[1]
    yc = line[2] * shape_im[0]
    w = line[3] * shape_im[1]
    h = line[4] * shape_im[0]
    print(xc-w/2, xc+w/2, yc-h/2, yc+h/2)
    label = 'goal' if line[0] else 'human'
    cv2.rectangle(im, (int(xc-w/2), int(yc-h/2)), (int(xc+w/2), int(yc+h/2)), (0, 255, 0), thickness=1, lineType=8, shift=0)
    # cv2.rectangle(im, (353, 215), (377, 265), (255, 0, 0), thickness=1, lineType=8, shift=0)
    cv2.putText(im, label, (int(xc-w/2), int(yc-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
cv2.imshow('labeled', im)
cv2.waitKey(0)