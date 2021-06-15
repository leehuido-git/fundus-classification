'''
classifier.py
Author: HUIDO LEE (j3jjj2021@naver.com)
normal medium(intermediate) abnormal
EfficientNetB0, EfficientNetB7
'''
import cv2
import numpy as np
from classifier_train import outside_masking, load_imgs_dir, resize, model, load_data, pred_1st_img
from tqdm import tqdm
from os.path import join, basename, isdir
from os import mkdir, getcwd

if __name__ == '__main__':
  local_path = getcwd()

  ########################################전처리
  print('-'*30)
  print('preprocessing data')
  input_path = join(local_path, 'input')
  pred_input_path = join(local_path, 'pred_input')
  gray_threshold = 50
  pred_1st_img(input_path=input_path, outputpath=pred_input_path, gray_threshold=gray_threshold)

  ########################################load data
  print('-'*30)
  print('load data')
  x, names = load_data(local_path, type='test', shuffle='False')
  print('input imgs len : {}'.format(len(x)))

  ########################################Define model
  print('-'*30)
  print('Create model')
  model = model(local_path, lr = 2e-5)

  ########################################start testing
  print('-'*30)
  print('Testing start')
  filename = join(local_path, 'checkpoint', 'checkpoint.ckpt')
  try:
    model.load_weights(filename)
  except:
    print('{}이 없습니다.'.format(filename))

  step = 10
  result = "image name, classifier, Accuarcy\n"
  for i in tqdm(range(len(x))):
    height, width, _ = x[i].shape
    input_imgs = []
    img_result = []
    for k in range(0, 359, step):
      matrix = cv2.getRotationMatrix2D((width/2, height/2), k, 1)
      input_imgs.append(cv2.warpAffine(x[i], matrix, (width, height)))
    for j in range(len(input_imgs)):
      input = np.expand_dims(input_imgs[j], axis=0)
      pred = model.predict(input)
      img_result.append(0 if np.argmax(pred[0])==0 else 1 if np.argmax(pred[0])==1 else 2)
    max_value = max(img_result.count(0), img_result.count(1), img_result.count(2))
#    result += '{}, {}, {:.2f}\n'.format(names[i], "normal" if img_result.count(0)==max_value else "abnormal" if img_result.count(1)==max_value else "medium", max_value/36)
    result += '{}, {}, {:.2f}\n'.format(names[i], "normal" if img_result.count(0)==max_value else "abnormal" if img_result.count(1)==max_value else "intermediate", max_value/36)

  with open(join(local_path,'output', 'result.csv'), 'w') as f:
    f.write(result)
    f.close()
  print('done.')