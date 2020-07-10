'''predict image'''
import csv
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

correct = 0
with open('car.csv', newline = '') as csvfile:
  reader = csv.reader(csvfile)
  y = [row for row in reader]
  
y_test = np.array(y, dtype=np.int32)

from keras.models import load_model
output_fn = 'CIFAR-10'
model = load_model(output_fn+'.h5')

for i in range(100):
  img = Image.open('car/'+str(i)+'.png')
  img_convert_ndarray = np.array(img)
  ndarray_convert_img = np.array(img)
  x_test = img_to_array(ndarray_convert_img)
  x_test = x_test.reshape(1, 32, 32, 3).astype('float32')
  x_test /= 255
  pred = model.predict_classes(x_test,verbose=0)
  true_flag = y_test[i]-pred
  if(true_flag == 0):
    correct = correct+1

print ('acc=', correct / 100)
