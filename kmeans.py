import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles


################################
######## FUNCTIONS HERE  #######
################################

def generateData(n):
    data = np.random.uniform(-10,10,(n,2))
    zeros_column = np.zeros((data.shape[0], 1))
    data_with_zeros = np.hstack((data, zeros_column))
    return data_with_zeros

def generateTestData():
    X1, Y1 = make_blobs(n_features=2, centers=5)
    data = X1
    zeros_column = np.zeros((data.shape[0], 1))
    data_with_zeros = np.hstack((data, zeros_column))

    # data = 1 ## WRITE DATA POINTS S.T. IT SEEMS TO BE CLUSTERED FAIRLY WELL
    return data_with_zeros

def findRandom(data,k):
    rand_idx = np.random.choice(data.shape[0], size=k, replace=False)
    k_centers = data[rand_idx]
    return k_centers

def findFarthest(data, k):
    n_points = data.shape[0]
    k_centers = []
    first_center_idx = np.random.choice(n_points)
    k_centers.append(data[first_center_idx])

    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(x - center) for center in k_centers) for x in data])
        next_center_idx = np.argmax(distances)
        k_centers.append(data[next_center_idx])
    ## WRITE LOGIC HERE!
    return k_centers

def findKMeansPlusPlus(data, k):

    ## WRITE LOGIC HERE!
    return k_centers

def findClosest(data,k_centers):
    for i in range(len(data)):
        dist = []
        datax = data[i][0]
        datay = data[i][1]
        for j in range(k):
            #1 refers to current point, 2 refers to a k_center point
            # distance = sqrt(sq(x2-x1) + sq(y2-y1))
            distance = np.sqrt(np.square(k_centers[j][0] - datax) + np.square(k_centers[j][1] - datay))
            dist.append(distance)
            
        
        # find min value/index in dist. That refers to the closest point
        #add index of closest k_centers point to the 3rd column of data
        data[i][2] = np.argmin(dist)
    return data

def computeCenters(data,k_centers):
    identifiers = data[:,2]
    unique_ids = np.unique(identifiers)
    for i, uid in enumerate(unique_ids):
        mask = (identifiers == uid)
       # newdata = data[mask]
       # newdata[:,1]
        k_centers[i,0] = np.mean(data[mask][:,0])
        k_centers[i,1] = np.mean(data[mask][:,1])
    return k_centers


def plotData(data, fignum):
    x = data[:,0]
    y = data[:,1]
    plt.figure(fignum)

    # # Create a plot
    plt.scatter(x, y)

    # # Add title and labels
    plt.title("Simple Scatter Plot")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    # # Display the plot
    plt.show()

def plotDataKMeans(data,k_centers,fignum):
    plt.figure(fignum)
    x = data[:,0]
    y = data[:,1]
    identifiers = data[:,2]
    unique_ids = np.unique(identifiers)
    colors = plt.cm.rainbow(np.linspace(0,1,len(unique_ids)))
    #colors = px.colors.qualitative.Plotly[:len(unique_ids)]    
    
    for i, uid in enumerate(unique_ids):
        mask = (identifiers == uid)
        plt.scatter(x[mask], y[mask], color=colors[i], label=f'ID {int(uid)}')
    plt.scatter(k_centers[:,0],k_centers[:,1],color="black",marker="x",label="Centroid")
    plt.legend()
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Scatter Plot by Identifiers')
    plt.show()


################################
######## START MAIN HERE #######
################################

n = 100 # Generate n points of data, uniformly between -10 and 10
data = generateData(n)
#data = generateTestData()
plotData(data, 1)

## Try Running K-Means Algorithm with LLoyd's Algorithm
# Step 1: Randomly pick k centers {u1, ..., uk}
# Step 2: Assign each point in the dataset to its closest center
# Step 3: Compute the new centers as the means of each cluster
# Step 4: Repeat 2 & 3 until convergence

#Step 1
k = 3

k_centers = findRandom(data,k)
print(k_centers)
k_centers0 = k_centers

#Step 2
data = findClosest(data,k_centers)
plotDataKMeans(data,k_centers,2)

# Step 3
k_centers = computeCenters(data,k_centers)
plotDataKMeans(data,k_centers,3)

k_centers1 = k_centers

#Now Iterating
data = findClosest(data,k_centers)
plotDataKMeans(data,k_centers,4)
k_centers = computeCenters(data,k_centers)
plotDataKMeans(data,k_centers,5)

k_centers2 = k_centers

data = findClosest(data,k_centers)
plotDataKMeans(data,k_centers,6)
k_centers = computeCenters(data,k_centers)
plotDataKMeans(data,k_centers,7)

k_centers3 = k_centers
