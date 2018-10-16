import matplotlib.pyplot as plt
import numpy as np 

data = np.loadtxt('history.log',skiprows=1,delimiter=',')

epoch = data[:,0]
acc = data[:,3]
loss = data[:,4]

plt.plot(epoch,acc,'r')
plt.plot(epoch,loss,'b')

plt.show()