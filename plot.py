from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import math



def dist(x, y, x_0, y_0):
    return math.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)

def interpolate(x, y, z):

    Z = []

    for i in range(len(x)):
        line = []
        for j in range(len(y)):
            print(str(i) + " " + str(j))
            if i == j:
                line.append(z[i])  
            else:
                min = 0
                d_min = dist(x[i], y[j], x[0], y[0])

                for n in range(len(x)):
                    d = dist(x[i], y[j], x[n], y[n])
                    if d < d_min:
                        d_min = d
                        min = n

                line.append(z[n])

        Z.append(line)

    return Z


def plot_implied_vol_surface():
  
    f = open("data.csv", "r")

    f.readline() 
    f.readline() 
    f.readline() 
    line = f.readline() 

    X = []
    Y = []
    Z = []

    while line:

        arr = line.strip().split(",")

        X.append((time.mktime(datetime.datetime.strptime(arr[0], "%m/%d/%Y").timetuple()) - time.time()) / (365 * 24 * 3600))
        Y.append(float(arr[8]))
        Z.append(float(arr[7]))

    
        line = f.readline() 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("T")
    ax.set_ylabel("Delta")
    ax.set_zlabel("IV")
    ax.scatter(np.array(X), np.array(Y), np.array(Z), color='b')
    #ax.plot_surface(np.array(X), np.array(Y), np.array(interpolate(X, Y, Z)), color='b')
    plt.show()


plot_implied_vol_surface()
