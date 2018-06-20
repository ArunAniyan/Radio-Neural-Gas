## Growing Neural Gas Fitting on Images
# Arun Aniyan
# SKA SA/ RATT
# arun@ska.ac.za
# 11-8-15


#-----------------------------------------#
# This code works on only gray scale images
#-----------------------------------------#


import sys
import time
import mdp
import numpy as np 
from scipy.misc import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import time

# Read input arguments 

arg = sys.argv 
infile = arg[1]

starttime = time.time()

# Load Image
image = imread(infile)

#image = rgb2gray(image)
image = np.squeeze(image)


#image = np.loadtxt(infile,delimiter=',')
# Initialize GNG graph model
gng = mdp.nodes.GrowingNeuralGasNode(max_nodes=2000, input_dim=3,lambda_=40)

# Convert Image to float  - GNG model reads only float data
float_data = np.float32(image)

# Apply Threshold
if len(sys.argv)>2:
	threshold = int(float(arg[2]))
	temp = float_data > threshold
	
	
else:
	temp = float_data

xdim = float_data.shape[0]
ydim = float_data.shape[1]

data = np.zeros((xdim*ydim,3))
binary_data = np.zeros((xdim*ydim,3))
i = 0
for x in range(xdim):
    for y in range(ydim):
        data[i][0] = x
        data[i][1] = y
        data[i][2] = float_data[x,y]

        binary_data[i][0] = x
        binary_data[i][1] = y
        binary_data[i][2] = temp[x,y]
        i += 1


# Delet Data points which below threshold
if len(sys.argv)>1:
	idx = np.where(binary_data[:,-1] != 1)
	data = np.delete(data,idx,axis=0)
	#data = data[:,0:-1]

# Shuffle Data
np.random.shuffle(data)
 
# Train Model 
gng.train(data)

objs = gng.graph.connected_components()
n_obj = len(objs)

print 'Number of connected components : ',n_obj
print 'Number of nodes : ',len(gng.graph.nodes)


plt.subplot(1,2,1)
plt.imshow(image,cmap='gray',aspect='equal')

plt.subplot(1,2,2)
plt.imshow(image,cmap='gray',aspect='equal')

fills = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff', '#ff0088', '#ff8800', '#0088ff']
color = range(n_obj)
for j,obj in enumerate(objs):
	
	for node in obj:
		fx, fy, fz = node.data.pos
		
		plt.scatter(np.round(fy),np.round(fx),s=20,c=fills[j % 8])

'''
plt.subplot(1,3,3)
plt.imshow(temp,cmap='gray')
'''

end  = time.time()

print "Total time is :", end - starttime
plt.show()







