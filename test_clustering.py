import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import read_storm_data


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    X3 = X[label == 3, :]
    X4 = X[label == 4, :]
    X5 = X[label == 5, :]
    X6 = X[label == 6, :]
    X7 = X[label == 7, :]
    X8 = X[label == 8, :]
    X9 = X[label == 9, :]
    X10 = X[label == 10, :]
    X11 = X[label == 11, :]
    X12 = X[label == 12, :]
    X13 = X[label == 13, :]
    X14 = X[label == 14, :]

    
    plt.plot(X0[:, 1], X0[:, 0], 'b', markersize = 4, alpha = .8)
    plt.plot(X1[:, 1], X1[:, 0], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 1], X2[:, 0], 'rs', markersize = 4, alpha = .8)
    plt.plot(X3[:, 1], X3[:, 0], 'ms', markersize = 4, alpha = .8)
    plt.plot(X4[:, 1], X4[:, 0], 'ys', markersize = 4, alpha = .8)
    plt.plot(X5[:, 1], X5[:, 0], 'cs', markersize = 4, alpha = .8)
    plt.plot(X6[:, 1], X6[:, 0], 'rs', markersize = 4, alpha = .8)
    plt.plot(X7[:, 1], X7[:, 0], 'ys', markersize = 4, alpha = .8)
    plt.plot(X8[:, 1], X8[:, 0], 'go', markersize = 4, alpha = .8)
    plt.plot(X9[:, 1], X9[:, 0], 'ms', markersize = 4, alpha = .8)
    plt.plot(X10[:, 1], X10[:, 0], 'gs', markersize = 4, alpha = .8)
    plt.plot(X11[:, 1], X11[:, 0], 'bs', markersize = 4, alpha = .8)
    plt.plot(X12[:, 1], X12[:, 0], 'rs', markersize = 4, alpha = .8)
    plt.plot(X13[:, 1], X13[:, 0], 'bs', markersize = 4, alpha = .8)
    plt.plot(X14[:, 1], X14[:, 0], 'gs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
"""
X = []

path = 'G:\\Lab\\storm_data\\storm_data.txt'
f = open(path, 'r')
f_read = f.read();
f_read = f_read.split('\n')

for i in range(len(f_read)/2):
	X.append([f_read[2*i], f_read[2*i+1]])

X = np.asarray(X)
kmeans = KMeans(n_clusters=15, random_state=0).fit(X)
print ('Centers found by scikit-learn:')
clusters = kmeans.cluster_centers_
print (clusters)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)
"""

path_folder = 'G:\\Lab\\storm_data\\cluster_test'

for i in range(1,94):
	path = path_folder + '\\'  + str(i) + '.txt'
	
	#X = []
	
	#f = open(path, 'r')
	#f_read = f.read();
	#f_read = f_read.split('\n')
	
	
	#for j in range(len(f_read)/2):
	#	X.append([f_read[2*j], f_read[2*j+1]])
	
	X = read_storm_data.read_data(i, path_folder)
	
	X = np.asarray(X)
	kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
	print ('Centers found by scikit-learn:')
	clusters = kmeans.cluster_centers_
	print (str(i))
	print (clusters)
	pred_label = kmeans.predict(X)
	kmeans_display(X, pred_label)
	
	#f.close()
	
	
#########################################
"""
import os
import shutil
src_path = 'G:\\Lab\\storm_data\\cluster_test'
dist_path = 'G:\\Lab\\storm_data\\test'
file_count = 1;

for filename in os.listdir(src_path):
	shutil.copy(src_path + '\\' + filename, dist_path + '\\' + str(file_count) + '.txt')
	file_count = file_count + 1


"""



	