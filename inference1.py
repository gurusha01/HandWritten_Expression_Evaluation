import tensorflow
import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv
model = keras.models.load_model('final.h5')
import os,sys

def trisect(img):
  image = cv2.imread(img)
  #plt.imshow(image)
  image_dash = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  data = np.asarray(image_dash)
  c = data.shape[1]//3
  p1 = data[:,:c]
  p2 = data[:,c : 2*c]
  p3 = data[:,2*c:]
  images = [p1,p2,p3]
  new_imgs =[]
  for img in images:
    img =np.pad(img,((13,14),(13,14)),'maximum')
    new_imgs.append(img)
  return new_imgs

def calc(op,n1,n2):
  op = int(op)
  n1 = int(n1)
  n2 = int(n2)
  if op == 10:
    return n1 + n2
  elif op == 11:
    return n1 - n2
  elif op == 12 :
    return n1*n2
  else:
    return n1//n2

def eval(img):
  images = trisect(img)
  #plt.imshow(images[0])
  ans = []
  for image in images:
    x = np.expand_dims(image, axis=2)
    #print(x.shape)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    y = model.predict(x)
    ans.append(np.argmax(y))
  #print(ans)
  if ans[0] >9 :
    return 'prefix'
  elif ans[1] > 9 :
    return 'infix'
  elif ans[2] > 9 :
    return 'postfix'
  else:
     return 'F'


#print(eval('SoML-50/data/13.jpg'))
if __name__ == '__main__':
    in_dir = sys.argv[1]
    out = csv.writer(open('jaishreeRAM_1.csv', 'a'))
    out.writerow(['Image Name','Label'])
    #print(os.listdir(in_dir)[7474])
    for images in os.listdir(in_dir):
      #print(eval(in_dir+'/'+images))
      row = [images,eval(in_dir+'/'+images)]
      out.writerow(row)






