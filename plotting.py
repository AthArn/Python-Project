import matplotlib.pyplot as pl
import numpy as nm
data = nm.loadtxt('history.log',skiprows=1,delimiter=',')
epoch = data[:,0]
acc = data[:,3]
loss = data[:,4]
pl.plot(epoch,acc,'r')
pl.plot(epoch,loss,'b')
pl.show()
