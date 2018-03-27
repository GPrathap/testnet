



import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
img = cv2.imread('golfcourse77.jpg')

h = np.zeros((300, 256, 3))

bins = np.arange(256).reshape(256, 1)
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

for ch, col in enumerate(color):
    hist_item = cv2.calcHist([img], [ch], None, [256], [0, 255])
    cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    pts = np.column_stack((bins, hist))
    cv2.polylines(h, [pts], False, col)

h = np.flipud(h)

cv2.imshow('colorhist', h)
cv2.waitKey(0)

#reconstructed_signal = SingularSpectrumAnalysis(df2[index:window_size+index], 16, True).execute(2)
#plt.hist(img.ravel(),256,[0,256])
plt.show()






