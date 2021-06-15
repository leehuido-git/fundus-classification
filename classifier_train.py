'''
classifier_train.py
Author: HUIDO LEE (j3jjj2021@naver.com)
normal medium(intermediate) abnormal
EfficientNetB0, EfficientNetB7
'''
from unicodedata import name
from numpy.core.defchararray import lower
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import msvcrt as m
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.backend import arange
from tqdm import tqdm
from os.path import isdir, join, splitext, basename
from os import walk, mkdir, getcwd

from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.python.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import layers

ext_list = ['.jpg', '.JPG', '.png', '.PNG']
labels = ['normal', 'abnormal', 'medium']

def wait_key():
  m.getch()

def load_imgs_dir(src_path):
  images = []
  if isdir(src_path):
    for (path, _, files) in walk(src_path):
      for filename in files:
        ext = splitext(filename)[-1]
        if (ext in ext_list):
          images.append(join(path, filename))
    if len(images) < 1:
      print('no file to process')
      exit()
    return images
  else:
    print('no folder to process')
    exit()

def outside_masking(input_img, x, y, r, choice):

  t = round(r/10)
  mask = np.zeros_like(input_img)
  cv2.circle(mask, (x, y), r-t, (255, 255, 255), -1)
  masked = cv2.bitwise_and(input_img, mask)
  if choice == True:
    mask = np.zeros_like(input_img)
    cv2.circle(mask, (x, y), 0, (255, 255, 255), 2*r-t)
    masked = cv2.bitwise_or(masked, mask)
    cv2.circle(masked, (x, y), 0, (0, 0, 0), 2*r-t)
  return masked[max(0, y-(r-t)):min(y+(r-t), input_img.shape[0]), max(0, x-(r-t)):min(x+(r-t), input_img.shape[1])]

def resize(input_img, size):
  return cv2.resize(input_img, dsize=(size, size))

def pred_1st_img(input_path, outputpath, gray_threshold):
  try:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_dir = load_imgs_dir(input_path)

    for j in tqdm(range(len(imgs_dir))):
      img = cv2.imread(imgs_dir[j])
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      mark = np.copy(gray_img)
      thresholds = (gray_img[:,:] > gray_threshold)
      mark[thresholds] = 255
      mark = cv2.GaussianBlur(mark, (5,5), 0)
      radius = min(np.shape(mark)[0],np.shape(mark)[1])
      circles = cv2.HoughCircles(mark, cv2.HOUGH_GRADIENT, 1, radius, param1=10,param2=100,minRadius=radius//3, maxRadius = radius//2)
      if circles is not None:
        if int(circles[0][0][0]) != 0:
          cut_img = outside_masking(gray_img, int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2]), False)
          if cut_img.shape[0] != 0 and cut_img.shape[1] != 0:
            resize_img = resize(cut_img, 224)
            resize_img = clahe.apply(resize_img)
            cv2.imwrite(join(outputpath, basename(imgs_dir[j])), resize_img)
  except Exception as e:
    print(e)

def pred_2nd_img(path, step = 10):
  try:
    imgs_dir = load_imgs_dir(path)
    for j in tqdm(range(len(imgs_dir))):
      img = cv2.imread(imgs_dir[j])
      height, width, _ = img.shape

      for k in range(step, 359, step):
        matrix = cv2.getRotationMatrix2D((width/2, height/2), k, 1)
        dst = cv2.warpAffine(img, matrix, (width, height))
        cv2.imwrite(join(path, basename(imgs_dir[j])[:-4]+'_'+str(k)+basename(imgs_dir[j])[-4:]), dst)
  except Exception as e:
    print(e)    

def load_data(local_path, shuffle, type):
  x, y, names = [], [], []
  if type == 'train':
    for i, label in enumerate(labels):
      pred_train_path = join(local_path, 'pred_train', label)
      imgs_dir = load_imgs_dir(pred_train_path)
      for j in tqdm(range(len(imgs_dir))):
        x.append(cv2.imread(imgs_dir[j]))
        y.append(i)
    y = tf.keras.utils.to_categorical(y)
    if shuffle:
      idx = np.arange(len(x))
      random.shuffle(idx)
      x = np.array(x)[idx]
      y = y[idx]
    return x, y
  elif type == 'test':
    pred_input_path = join(local_path, 'pred_input')
    imgs_dir = load_imgs_dir(pred_input_path)
    for j in tqdm(range(len(imgs_dir))):
      x.append(cv2.imread(imgs_dir[j]))
      names.append(basename(imgs_dir[j])[:-4])
    return x, names

def model(local_path, lr=2e-5):
  input_shape=(224,224,3)
#  conv_base = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
  conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

  model = models.Sequential()
  model.add(conv_base)
  model.add(layers.GlobalMaxPool2D(name='maxpool_flatten'))
  model.add(layers.Dropout(0.5, name="dropout_out_"+str(0.5)))
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dropout(0.2, name="dropout_out_"+str(0.2)))
  model.add(layers.Dense(3, activation='softmax'))
  model.summary()
  tf.keras.utils.plot_model(model, to_file=join(local_path, 'model.png'), show_shapes=True, show_layer_names=True)

  model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(lr=lr),
    metrics=["acc"])
  return model

##############################################GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    print(len(gpus))
    if(len(gpus) == 1):
      # 첫 번째 GPU만 사용하도록 제한
      tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    else:
      list([ '/gpu:'+str(i) for i in arange(len(gpus))])
      mirrored_strategy = tf.distribute.MirroredStrategy(devices=list(['/gpu:'+str(i) for i in range(len(gpus))]))
      tf.compat.v1.disable_eager_execution()
  except RuntimeError as e:
    print(e)

if __name__ == '__main__':
  #Check input parameters
  if len(sys.argv) is not 3:
    exit('usage: python {} <preprocessing(True/False)> <testing(True/False)>'.format(sys.argv[0]))
  
  preprocessing = sys.argv[1]
  test = sys.argv[2]

  ##############################################디렉토리 트리 확인, 만듦
  local_path = getcwd()

  tree = [join(local_path, 'train'), join(local_path, 'train', labels[0]), join(local_path, 'train', labels[1]), join(local_path, 'train', labels[2])\
    , join(local_path, 'pred_train'), join(local_path, 'pred_train', labels[0]), join(local_path, 'pred_train', labels[1]), join(local_path, 'pred_train', labels[2])\
    , join(local_path, 'input'),join(local_path, 'pred_input'), join(local_path, 'output')]
  tree_result = []
  tree_result = list([ True  if  isdir(i) else False for i in tree ])
  for i, dir in enumerate(tree):
    if tree_result[i] is False:
      mkdir(tree[i])

  ########################################전처리
  if preprocessing == 'True':
    print('-'*30)
    print('preprocessing data')
    gray_threshold = 50

    for i in labels:
      train_path = join(local_path, 'train', i)
      pred_train_path = join(local_path, 'pred_train', i)
      pred_1st_img(input_path=train_path, outputpath=pred_train_path, gray_threshold=gray_threshold)
      print("Please check {} data".format(pred_train_path))
      wait_key()
      pred_2nd_img(path=pred_train_path, step=10)

  ########################################load data
  print('-'*30)
  print('data split')
  test_split = 0.1
  x, y = load_data(local_path, shuffle=True, type='train')
  train_x, train_y = x[:-int(len(x)*test_split)], y[:-int(len(x)*test_split)]
  test_x, test_y = x[-int(len(x)*test_split):], y[-int(len(x)*test_split):]
  print('train len : {}, test len : {}'.format(len(train_x), len(test_x)))

  ########################################Define model
  print('-'*30)
  print('Create model')
  model = model(local_path, lr = 2e-5)

  ########################################start training
  print('-'*30)
  print('Training start')
  filename = join(local_path, 'checkpoint', 'checkpoint.ckpt')
  earlystopping = EarlyStopping(monitor='val_loss', patience=3)
  checkpoint = ModelCheckpoint(
    filename, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose = 1)
  hist = model.fit(train_x, train_y, 32, epochs=100, shuffle=True, validation_split=0.3, callbacks=[checkpoint, earlystopping])  
#  hist = model.fit(train_x, train_y, 4, epochs=100, shuffle=True, validation_split=0.3, callbacks=[checkpoint, earlystopping])  
  # Loss History
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.plot(hist.history['acc'])
  plt.plot(hist.history['val_acc'])
  plt.title('model loss')
  plt.ylabel('rate')
  plt.xlabel('epoch')
  plt.legend(['loss', 'val loss', 'acc', 'val acc'], loc='upper right')
  if test == 'False':
    plt.savefig(join(local_path, 'loss.png'))
    plt.show()  
  if test == 'True':
    """
    ########################################Define model
    print('-'*30)
    print('Create model')
    model = model(lr = 2e-5)

    ########################################start testing
    print('-'*30)
    print('Testing start')
    filename = join(local_path, 'checkpoint', 'checkpoint.ckpt')
    try:
      model.load_weights(filename)
    except:
      print('{}이 없습니다.'.format(filename))
    """ 
    pred = model.predict(test_x)
    pred_result = list(np.argmax(pred[i]) for i in range(len(pred[:])))
    test_y = list(np.argmax(test_y[i]) for i in range(len(test_y)))
    test_result = 0
    for i in range(len(pred_result)):
      if pred_result[i] == test_y[i]:
        test_result = test_result + 1
    print('test imgs = {}'.format(len(test_y)))
    print('test img acc = {}'.format((test_result/len(test_y))*100))
    plt.text(1, max(hist.history['loss'])-0.5, 'test imgs = {} \ntest img acc = {:.2f}'.format(len(test_y), (test_result/len(test_y))*100), horizontalalignment='left')
    plt.savefig(join(local_path, 'loss.png'))
    plt.show()  

