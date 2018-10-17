import numpy as nm
import keras
from keras.layers import Dense,Flatten,Dropout,Input
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,CSVLogger
import os
labels = nm.genfromtxt(r'./data/train.csv',delimiter = ',',skip_header = 1,skip_footer = 0, usecols = 0, dtype = nm.int32)
train = nm.genfromtxt(r'./data/train.csv',delimiter=',',skip_header=1,skip_footer = 0, usecols = range(1,785))
labels = nm.eye(nm.max(labels)+1)[labels]
train = nm.array([data.reshape(28,28,1) for data in train])
os.system('rmdir /S /Q "{}"'.format('./models/'))
os.makedirs('{0}/'.format('./models/'))
train_gen = ImageDataGenerator(zoom_range = 0.22,rotation_range = 35,width_shift_range = 0.22,height_shift_range = 0.22)
inp = Input(shape = (28,28,1), dtype = 'float32')
model = Conv2D(16,(3,3),padding='same',activation='relu')(inp)
model = MaxPooling2D((2,2))(model)
model = Dropout(0.3)(model)
model = Conv2D(32,(3,3),padding='same',activation='relu')(model)
model = MaxPooling2D((2,2))(model)
model = Conv2D(64,(3,3),padding='same',activation='relu')(model)
model = MaxPooling2D((2,2))(model)
model = Flatten()(model)
model = Dropout(0.2)(model)
output = Dense(10,activation='softmax')(model)
model = Model(inputs = inp,outputs = output)
model.summary()
train_generator = train_gen.flow(train[:40000],labels[:40000],batch_size = 128)
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
checkpt = ModelCheckpoint(filepath = r'./models/model.{epoch:02d}-{val_loss:.2f}.hdf5',monitor='val_loss',save_best_only=True)
csv_logger = CSVLogger('history.log')
model.fit_generator(train_generator,steps_per_epoch=int(40000/128),epochs=3000,callbacks=[csv_logger,checkpt],validation_data = (train[40000:],labels[40000:])) 
