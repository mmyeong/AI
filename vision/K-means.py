import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from sklearn.cluster import KMeans
#read image
img = cv2.imread('need/ROI.jpg')

#convert from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixels = img.shape[0]*img.shape[1]
channels = 3
data = np.reshape(img[:, :, :channels], (pixels, channels))

histo_rgb, _ = np.histogramdd(data, bins=256)
r, g, b = np.nonzero(histo_rgb)

clt = KMeans(n_clusters=3)  # k개의 데이터 평균을 만들어 데이터를 clustering하는 알고리즘
clt.fit(data)
n1 = clt.cluster_centers_  # 각 클러스터의 중심 위치
print(n1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#get rgb values from image to 1D array
#r, g, b = cv2.split(img)
r = r.flatten()
g = g.flatten()
b = b.flatten()

#plotting
#fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(r, g, b)
scatter = ax.scatter(
            n1[0],
            n1[1],
            n1[2],
            s = 500,
            marker='o',
            c=['red', 'orange', 'yellow'],
            label = "Centroid"
            )
plt.show()