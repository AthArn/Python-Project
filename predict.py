import keras 
import numpy as nm
import pandas as pd
model = keras.models.load_model(r'./models/model.06-0.09.hdf5') #give name of model in /models/ here
test = nm.genfromtxt(r'./data/test.csv',delimiter=',',skip_header=1,skip_footer = 0, usecols = range(0,784))
test = nm.array([data.reshape(28,28,1) for data in test])
pred = model.predict(test)
pred = nm.argmax(pred,axis=1)
print(pred.shape)
sample_submission_df = pd.read_csv(r'.\data\sample_submission.csv')
my_submission_df = pd.DataFrame()
my_submission_df['ImageId'] = sample_submission_df.ImageId
my_submission_df['Label'] = pred
my_submission_df.to_csv('my_submission.csv', index=False)
