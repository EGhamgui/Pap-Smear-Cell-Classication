########### This file contains all the functions used in the notebook ###########



# Import libraries for the functions below
import os
import gc 
import cv2
import time
import shutil
import imageio
import numpy as np 
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from skimage import measure
from skimage import feature
import matplotlib.pyplot as plt 
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RF 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef
from keras import backend as K
import matplotlib
from keras.preprocessing.image import ImageDataGenerator



# Function to plot the structure of the dataset
def draw_pie(dataset, column, title):
    """
    Draw a pie chart for the count of each class 
    Input : 
        - dataset: The dataset 
        - column: the column of the target to count its value 
        - title: the title of the plot
        
    """ 
    
    # Define the background form
    sns.set()
    
    # Plot PieChart
    dataset.plot.pie(y=column,figsize=(10,7),autopct='%1.1f%%', startangle=-90)
    
    # Set the legend and title
    plt.legend(loc="upper right", fontsize=12)
    plt.title(title,fontsize=16)
    plt.show()

##########################################################

def plot_images ( path  , nbr, Classes, Class ):
  """
  This function plot images from the data 
  Input: 
        - Path : path to data 
        - nbr : the number of images to plot
        - classes : the classes
        - class : the class to plot 
  
  """

  ax = plt.subplots (1, nbr,figsize=(20,15))

  # Extract the images to plot
  n = len(Classes[Class].unique())
  p = n*nbr
  k = 0
  for i in sorted(Classes[Class].unique()):
    im_name = Classes.loc[Classes[Class] == i].index[:nbr]
    im_name = [str(i) +'.bmp' for i in im_name]
    im_name = [i for i in os.listdir(path) if i in im_name]

    # Plot images 
    for j in range(nbr) :
      im1 = cv2.imread(path + im_name[j])
      plt.subplot(1, p, k+1)
      plt.imshow(im1)
      plt.title('Class : '+ str(i))
      k+=1

  plt.show()
    
##########################################################

def confusion_matrix_f (Y_t, Y_p):
  """
  Plot the confusion matrix in a heatmap
  Input: 
      - Y_t: the target 
      - Y_p: the predictions 
  
  """
  # Define the background form
  sns.set(font_scale = 1.1)
  
  # Confusion matrix 
  c_m=(100*confusion_matrix(Y_t, Y_p ,normalize='true')).round(2)

  # Plot the heatmap
  sns.heatmap(c_m, fmt='g',cmap = "Blues", annot = True, linewidths = 0.5 , cbar = False)
  plt.ylabel('Actual Values')
  plt.xlabel('Predicted Values')
  plt.title('Confusion Matrix (%)')
  plt.show()
  
##########################################################

def GetPath(Path, SetType):
  """
  Get the path, where to save the image
  Input:
    - Path : first path
    - SetType : the type of the folder 
  Output: 
    - path_to : the path sought

  """
    
  # First path
  path_to = Path + 'Data_masked/'
  
  # If the folder do not exist, we create it 
  # and we add its name to the path
  if not os.path.exists(path_to):
      os.mkdir(path_to)
  path_to += SetType + '/'
  
  # If the path exists, create the folder 
  if not os.path.exists(path_to):
      os.mkdir(path_to)
      
  return path_to

##########################################################

def SaveImg(Path, Classes, SetType, i, img):
  """
  Save the image in the correct path
  
  Input: 
      
      - Path : the original path  
      - Classes : the class of the image
      - SetType : the type of the folder 
      - i : the name of the image 
      - img : the image to save 

    """    

  # choose the target column
  column = 'ABNORMAL' if 'Binary' in Path else 'GROUP'
  
  # Find the path of the folder 
  path_to = GetPath(Path, SetType)
  
  # Split the data into folders of classes 
  if SetType == 'Train':
      f = str(Classes.loc[int(i.split('.')[0]) , column])
      path_to += f + '/'
      if not os.path.exists(path_to):
        os.mkdir(path_to)
  else:
    for f in Classes[column].unique().astype(str):
      path = path_to + f + '/'
      if not os.path.exists(path):
        os.mkdir(path)
    path_to += '0/'

  # Save the image 
  path_to += i.split('.')[0] + '.jpg'
  matplotlib.image.imsave(path_to, img.astype('uint8'))
  
##########################################################

def ImportSaveImages(OriginalPath, PathTo, Classes, SetType, n = None):
  """
  Save all images in the correct path
  Input: 
      - OriginalPath: original path
      - PathTo: path to save 
      - Classes: the classes  
      - SetType: type of data (Train or Test)
      - n : number of images to import
      
  Output:
      - Missing images (if exists)
      
  """
  
  # Create the folder 
  if not os.path.exists(PathTo):
    os.mkdir(PathTo)

  # Path of binary folder 
  PathBinary = PathTo + 'Binary/'
  if not os.path.exists(PathBinary):
    os.mkdir(PathBinary)
    
  # Path of multiclass folder 
  PathMulti = PathTo + 'Multi/'
  if not os.path.exists(PathMulti):
    os.mkdir(PathMulti)

  # Save the images of the ROI in the correct folder 
  # In this step: we need to import the images and masks 
  # Then, multiply the original images with the masks
  # to obtain the the ROI, these images will be then used in 
  # the deep learning classification task
  
  # The path of images
  PathImages = OriginalPath + SetType + '/'
  
  # The name of images
  imgs = sorted([i for i in os.listdir(PathImages) if 'seg' not in i])
  if n != None:
      imgs = imgs[:n]
  
  # Missed images 
  ToDo = []

  # Images Lists
  list_original_images = [] 
  list_segCyt_mask = []
  list_segNuc_mask = []
  
  # Impport images, detect the ROI,then save the new images 
  for i in imgs:
    try:
      img = imageio.imread(PathImages + i[:-4] + '.bmp')
      sn = imageio.imread(PathImages + i[:-4] + '_segNuc.bmp')
      sc = imageio.imread(PathImages + i[:-4] + '_segCyt.bmp')

      # Save original images and scaled masks 
      list_original_images.append(img)
      list_segCyt_mask.append(sc/255)
      list_segNuc_mask.append(sn/255)
      
      mask_both = img.copy()
      mask_both[:,:,0] *= (sn + sc)
      mask_both[:,:,1] *= (sn + sc)
      mask_both[:,:,2] *= (sn + sc)

      # Save the new images 
      Img = mask_both.copy()
      for Path in [PathBinary, PathMulti]:
        SaveImg(Path, Classes, SetType, i, Img)

    # In the case when the image is not saved, an exception rises and 
    # the name of the image is added to the list ToDo to be saved later 
    except Exception as ex:
      print(ex)
      ToDo.append([PathImages + i])
      
  if ToDo != []:
    print("Missing: ", ToDo)

  return imgs, list_original_images, list_segCyt_mask, list_segNuc_mask

##########################################################

def SetValPaths(PathTo):
  """
  Split the training set into train set and validation set
  Input:
      - PathTo: the path of the folder 
  
  """
  
  # Loop over the folder in the path 
  for c in [f for f in os.listdir(PathTo) if '.csv' not in f]:
    Path_class = PathTo + c + '/'
    
    # Dictionary that splits the data with the same percentages 
    N = {'Binary' : {0: 460, 1: 461},
         'Multi': {0: 204, 1: 165, 2: 20, 3: 31, 4: 38, 5: 30, 6: 41, 7: 173, 8: 170}}

    # Split the data in folder into train and validation datasets
    for i in os.listdir(Path_class):
      
      # Path to image 
      Path_im = Path_class + i + '/'
      
      # Path to validation dataset folder 
      Path_SetType = Path_im + 'Val/'
      if not os.path.exists(Path_SetType):
        os.mkdir(Path_SetType)
      else:
        os.system('rm -r ' + Path_SetType)
        os.mkdir(Path_SetType)
          
      
      # Path to training dataset folder 
      for g in os.listdir(Path_im + 'Train/'):
        Path_group = Path_SetType + g + '/'
        
        if not os.path.exists(Path_group):
          os.mkdir(Path_group)
          
        Path_group_train = Path_im + 'Train/' + g + '/'
        files = sorted(os.listdir(Path_group_train))[:N[c][int(g)]]
        
        # Move the images and save it in the right folder 
        for f in files:
            if f not in os.listdir(Path_group):
              shutil.move(Path_group_train + f , Path_group)
              

##########################################################

############ Features Extraction ############
def hog (img) : 
  """
  Create the Hog features
   
  Input: 
    - img: the original image
  Output:
    - hog: the value of HOG
    
  """
    
  hog = feature.hog(img, orientations=9,
                    pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    block_norm='L2-Hys', visualize=False, transform_sqrt=True)
  return hog

def HandCrafted(img , mask0, mask1):
  """
  Create Hand Crafted Features
  Input: 
       - img: the original image
       - mask0: the first mask 
       - mask1: the second mask 
  Output:
       - f: list  of features
       
  """
  
  # List of channels 
  c=['red','green','blue']
  
  # Initialize the list of features
  f = []

  # Loop over features 
  for k in range(2):
      
    # Loop over masks 
    mask = vars()['mask' + str(k)] 
    for i in range (len(c)): 

      # Extract the ROI
      image_1 = np.zeros(img.shape)
      for j in range(len(c)) :
        image_1[:,:,j] = img[:,:,j] * mask 
    
      # Extract the ROI for each channel
      img1 = image_1[:,:,i]

      # Feature1: average intencity
      vars()['avg_intensity'+ str(k) + '_' + c[i]] = np.mean ( img1 )
      f.append (vars()['avg_intensity'+ str(k) + '_' + c[i]])
      
      # Calculate the histogram 
      hist = cv2.calcHist([image_1.astype(np.float32)], [i], None, [256], [0, 256]).reshape(256) 
      hist = hist / np.sum(hist)

      # Feature2: the entropy
      vars()['entropy'+ str(k) + '_' + c[i]] = -np.sum(np.multiply(hist, np.log2(hist + 10**(-20))))
      f.append(vars()['entropy'+ str(k) + '_' + c[i]])

      # Feature3: elongation
      vars()['elong'+ str(k) + '_' + c[i]] = measure.inertia_tensor_eigvals (img1) [-1 ] / measure.inertia_tensor_eigvals (img1) [0]
      f.append(vars()['elong'+ str(k) + '_' + c[i]]) 

      # Feature4: compactness
      vars()['comp'+ str(k) + '_' + c[i]] = ( 1 /(4 * np.pi * np.sqrt(measure.inertia_tensor_eigvals(img1) [-1 ] * measure.inertia_tensor_eigvals(img1)[0]) ) ) 
      f.append(vars()['comp'+ str(k) + '_' + c[i]]) 

  # Feature5: N/C ratio 
  f.append(mask1.sum() / mask0.sum())

  # Feature6 and Feature7 : Orientation / distance between centers
      
  # Find contours for the nuclus
  cnt0 = measure.find_contours(mask0,0.5)
  
  # Find contours for the nuclus + cytoplasm
  cnt1 = measure.find_contours(mask0 + mask1,0.5)[0]
  
  # Calculate moments and center of nuclus and cytoplasm
  if len(cnt0)== 0 : 
    dist = np.nan
    orient0 = np.nan
    m1 = measure.moments(cnt1)
  else: 
    cnt0 = measure.find_contours(mask0,0.5)[0]
    m0 = measure.moments(cnt0)
    
    # Coordinate of the center of nuclus
    x0 = int (m0[1,0]/m0[0,0])
    y0 = int (m0[0,1]/m0[0,0])
    
    # Orientation of nuclus
    orient0 = (1/2) * np.arctan((2* m0[1,1])/(m0[2,0]-m0[0,2]))
    
    m1 = measure.moments(cnt1)
    
    # Coordinate of the center of cytoplasm
    x1 = int (m1[1,0]/m1[0,0])
    y1 = int (m1[0,1]/m1[0,0])
    
    # The distance between centers of nuclus and cytoplasm
    dist = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    f.append(dist)
  
  # Feature6: orientation of the cytoplasm
  orient1 = (1/2) * np.arctan((2* m1[1,1])/(m1[2,0]-m1[0,2]))
  f.append(orient0)
  f.append(orient1)
    
  return(f)

##########################################################

def GetFeatures(original_Train, segNuc_Train, segCyt_Train, names_Train, Classes, FeatType = 'HandCrafted'):
  """
  This function generate the feature 
  Input: 
      - Classes : data that contains the labels of each image
      - FeatType : features to create, it can be: HandCrafted or HOG
      - original_Train : Original images
      - segNuc_Train : masks of nuculs 
      - segCyt_Train : masks of cytoplasm
      - names_Train : names of images 
  Output:
      - X: features
      - Y_b : target of binary class
      - Y_m : target of multi class

  """
  # Calculate hand crafted features
  if FeatType == 'HandCrafted':
    Feats = pd.DataFrame()
    for i in range(len(original_Train)):
      Feats = Feats.append(pd.Series(HandCrafted(original_Train[i], segNuc_Train[i], segCyt_Train[i])),ignore_index=True)
      X = Feats

  # Calculate HOG features
  if FeatType == 'HOG':
    Feats = pd.DataFrame()
    for i in range(len(original_Train)):
      Feats = Feats.append(pd.Series(hog(original_Train[i])),ignore_index=True)
      X = Feats
    
  # Merge with target data for binary and multiclass target
  Y_b = Classes['ABNORMAL']
  Y_m = Classes['GROUP']

  labels=dict()
  labels1=dict()

  for e in names_Train:

      name=e[:-4]
      labels[name]=Y_b.loc[int(name)]
      labels1[name]=Y_m.loc[int(name)]

  Y_b = np.array(list(labels.values()))
  Y_m = np.array(list(labels1.values()))

  # Construct the new data 
  # Drop the None values 
  X['ABNORMAL'] = Y_b
  X['GROUP'] = Y_m
  X = X.dropna()

  Y_b = X['ABNORMAL']
  Y_m = X['GROUP']

  X = X.drop(columns=['ABNORMAL' ,'GROUP' ])

  return X, Y_b , Y_m

##########################################################

def TrainTestSplit(X, Y, scale = False, test_size = 0.25, random_state=0 , Type='binary'):
  """ 
  This function split the data into train and test data 
  After that, it performs the scaling of the data according to the parameter 
  <scale>

  If this value is true, we scale the data 

  - Type : for multiclass to make the bootstrap method

  """

  # Split the data 
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

  # Balance the data in multiclass case
  if Type == 'multi' :
    X_train['GROUP'] = y_train.values
    Max = len(X_train.loc[X_train['GROUP'] == 0])
    for i in range(1 , 9):
      n = Max // X_train['GROUP'].value_counts().loc[i]
      if n > 1:
        add_boot = X_train.loc[X_train['GROUP'] == i]
        index = []
        for i in range(n-1):
          index.append(add_boot.index.tolist())
          X_train = X_train.append(add_boot)
        del add_boot
        gc.collect()
    gc.collect()
    y_train = X_train['GROUP']
    X_train.drop(['GROUP'] , axis = 1 , inplace = True)

  # Scale the data 
  if scale == True:
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index = X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), index = X_test.index)
    
  # Rename the columns
  X_train.columns=X.columns
  X_test.columns=X.columns

  return X_train, X_test, y_train, y_test

##########################################################

# Models Functions
def Modeling (X_train, X_test, y_train, y_test, model, params_grid = {}, other_params = {}, Scoring = 'matthews_corrcoef', gridsearch = False):
  """
  This function groups all the modeling algorithms 
  Also, this function perform also the grid search in order to determine the best 
  parameters of the model. This is will help us to obtain better predictions 

  Input: 
        - X_train
        - X_test
        - y_test
        - y_train
        - model : the name of the model 
        - params_grid : parameters for the gridsearch 
        - other_params : initial parameters 
        - Scoring : scoring function 
        - gridsearch : {True or false} whether we use gridsearch or not 

  Output : 
        - vector of scores and predictions 

  """
  # Initialize the computational time 
  StartTime = time.time()

  # Chose the model 
  if model == 'SVM_Linear':
    mod = LinearSVC(**other_params)

  elif model == 'SVM':
    mod = SVC(**other_params)

  elif model == 'DecisionTree':
    mod = DTC(**other_params, random_state=0)

  elif model == 'RandomForest':
    mod = RF(**other_params)
    
  # Perform the grid search method 
  if gridsearch == True:
    grid = GridSearchCV(mod, params_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Print best scores and best parameters 
    print("Best training Score: {} %".format(round(100*grid.best_score_, 3)))
    print("Best training params: {}".format(grid.best_params_))
    estimator_best=grid.best_estimator_

  else:
    estimator_best = mod
  
  # Calcuate the different performance metrics for the best model 
  estimator_best.fit(X_train, y_train)
  y_pred_test = estimator_best.predict(X_test)
  y_pred_train = estimator_best.predict(X_train)

  # Calculate the computational time
  endTime = str(round(time.time() - StartTime, 3))

  # Calculate the scoring 
  ### Binary scoring 
  if Scoring == 'matthews_corrcoef':
    score_train = 100*matthews_corrcoef(y_train, y_pred_train)
    score_test = 100*matthews_corrcoef(y_test, y_pred_test)

  ### Multiclass scoring 
  else:
    score_train = 100*(y_train==y_pred_train).sum()/len(y_train)
    score_test = 100*(y_test==y_pred_test).sum()/len(y_test)

  # Print the different values 
  print(Scoring + ' for ' + model +  'Model; in-sample: ', round(score_train , 3) , '%')
  print(Scoring + ' for ' + model +  'Model; out-of-sample: ', round(score_test , 3) , '%')

  # Finally end up with plotting the confusion matrix
  cnf_matrix = confusion_matrix_f(y_pred_test, y_test)

  # Return the vector 
  ToReturn = [y_pred_train, y_pred_test, score_train, score_test, endTime]
  if gridsearch == True:
      ToReturn.append(grid.best_params_)

  # Print the computational time 
  print('Runtime to fit the model:', endTime, 'seconds')

  return tuple(ToReturn)

##########################################################

def SaveSubmission(OriginalPath, PathTo, Class, model, Classes):
  """
  This function generates the submission file in PathTo folder

  Input: 
        - PathTo: folder created at the beginning of the notebook
        - Class: ABNORMAL or GROUP
        - model: deep learning model fitted

  Output : 
        - predictions 

  """
  # Create a generator for the test data
  # This generator is only allowed to rescale images
  # and do not perform any other transformation

  ClassType = 'Binary' if Class == 'ABNORMAL' else 'Multi'
  ImageDataGenerator(rescale=1./255)

  # Import the test dataset  
  test_datagen = ImageDataGenerator(rescale=1./255)
  test_generator = test_datagen.flow_from_directory(PathTo + '/' + ClassType + '/Data_masked/Test/', shuffle=False)

  # Predict the data 
  predicted_data = Predict(test_generator, model, Class, Classes).drop([Class], axis = 1)
  data = pd.read_csv(OriginalPath + '/SampleSubmission.csv')
  predicted_data.index = predicted_data.index.astype(int)
  predicted_data = predicted_data.loc[predicted_data.index.isin(data['ID'].values)]

  # Save the values of the predictions
  predicted_data.to_csv(PathTo + '/SampleSubmission_' + ClassType + '.csv')

  return predicted_data

##########################################################

# Binary metric function with backed. It is used in deep learning evaluation 
def matthews_correlation(y_true, y_pred):
    """
    Metric to evaluate the performance, used in Deep Learning Models
    Input:
       - y_true: the target 
       - y_pred: the predictions 
    Output: 
       - The value of the matthews correlation
  
    """
    
    # Change the type of the input vectors
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    # Calculate the true positive metric
    tp = K.sum(y_pos * y_pred_pos)
    
    # Calculate the true negative metric
    tn = K.sum(y_neg * y_pred_neg)

    # Calculate the false positive metric
    fp = K.sum(y_neg * y_pred_pos)
    
    # Calculate the false negative metric
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

##########################################################
def Evaluation (X_train, X_test, y_train, y_test , model):
  """
  This function plot the impact of the parameters 
  on the model 

  Input : 
        - model : the name of the model 

  """
  fig , ax = plt.subplots(1, 2, figsize=(15,7))
    
  # min_samples_split
  TTest_ss=[]
  TTrain_ss=[]
    
  for i in range(2,15):

      # Fit the Model
      if model == 'DT':
        clf = DTC(min_samples_split=i, random_state=0)
      else: 
        clf = RF(min_samples_split=i, random_state=0)

      clf.fit(X_train,y_train)
      
      # Score Values 
      scoreTrain=clf.score(X_train,y_train)
      scoreTest=clf.score(X_test,y_test)
    
      # Save the scores 
      TTrain_ss.append(scoreTrain)
      TTest_ss.append(scoreTest)
    
  ax[0].plot(np.arange(2,15) , TTrain_ss, label='Training performance');
  ax[0].plot(np.arange(2,15) , TTest_ss,label='Test performance'); 
  ax[0].set_title('min_samples_split')
  ax[0].legend();
    
    
  # min_samples_leaf
  TTest_sl=[]
  TTrain_sl=[]
    
  for i in range(2,15):

      # Fit the Model
      if model == 'DT':
        clf1 = DTC(min_samples_leaf=i, random_state=0)
      else: 
        clf1 = RF(min_samples_leaf=i, random_state=0)

      clf1.fit(X_train,y_train)
    
      # Score Values 
      scoreTrain=clf1.score(X_train,y_train)
      scoreTest=clf1.score(X_test,y_test)
    
      # Save the scores
      TTrain_sl.append(scoreTrain)
      TTest_sl.append(scoreTest)
    
  ax[1].plot(np.arange(2,15),TTrain_sl,label='Training performance');
  ax[1].plot(np.arange(2,15),TTest_sl,label='Test performance');
  ax[1].set_title('min_samples_leaf')
  ax[1].legend();
    
  # Print Best Values 
  print("The value of min_samples_split that maximizes the training score is : ",TTrain_ss.index(max(TTrain_ss))+2)
  print("The value of min_samples_split that maximizes the test score is : ",TTest_ss.index(max(TTest_ss))+2)
  print("The value of min_samples_leaf that maximizes the training score is : ",TTrain_sl.index(max(TTrain_sl))+2)
  print("The value of min_samples_leaf that maximizes the test score is : ",TTest_sl.index(max(TTest_sl))+2)

#########################################################

def Predict(generator, model, Class, Classes):

  """
  This function is used to make predictions
  
  """

  # Predict the data 
  pred = model.predict(generator)
  
  # Retrieve images indexes 
  filenames = generator.filenames

  # Transform it into a list 
  nb_samples = len(filenames)

  # Then transform the list (nb_samples) into a dataset 
  predicted_data = pd.DataFrame(pred , index = filenames)

  # Turn predictions into binary labels
  if Class == 'ABNORMAL':
    predicted_data['Predict'] = 0
    predicted_data.loc[predicted_data[0] < predicted_data[1] , 'Predict'] = 1
  else:
    l = []
    for i in range(len(pred)):
      l.append(np.argmax(pred[i]))
    predicted_data['Predict'] = l

  # Match each value to its actual index as in the sample submission file
  predicted_data.index = predicted_data.index.str.split('/').str[1].str.split('.').str[0].astype(int)
  predicted_data = predicted_data[['Predict']]
  predicted_data.index.names = ['ID']

  return predicted_data.join(Classes[[Class]])

#########################################################

def Evaluation_DL(train_pred, val_pred, Class, Scoring):
  """
  This fuunction used to make evaluation for the deep learning 
  task and plot the confusion matrix 
  Input : 
        - train_pred : prediction and target on the training data 
        - val_pred : prediction and target on the validation data
        - class : tthe class 
        - scoring : the scoring metric

  """

  if Scoring == 'matthews_corrcoef':
    score_train = 100*matthews_corrcoef(train_pred[Class], train_pred['Predict'])
    score_test = 100*matthews_corrcoef(val_pred[Class], val_pred['Predict'])
  else:
    score_train = 100*(train_pred[Class] == train_pred['Predict']).sum() / len(train_pred)
    score_test = 100*(val_pred[Class] == val_pred['Predict']).sum() / len(val_pred)

  print(Scoring + ' for Deep Learning Model; in-sample: ', round(score_train , 3) , '%')
  print(Scoring + ' for Deep Learning Model; out-of-sample: ', round(score_test , 3) , '%')
  
  # Finally end up with plotting the confusiin metrix
  cnf_matrix = confusion_matrix_f(val_pred[Class], val_pred['Predict'])
  
##########################################################

def Boundaries( model, model1, X_train, X_test, y_train, y_test ):

  """
  This function study the feature importance 
  and plot the boundaries between the two first important features 

  Input : 
        - model : best model 

  """
  
  # Calculate the importance vectors form model 1
  model.fit(X_train, y_train)
  importances = model.feature_importances_
  indices = np.argsort(importances)[::-1]

  # Calculate the importance vectors form model 2
  model1.fit(X_train, y_train)
  importances1 = model1.feature_importances_
  names = X_train.columns
  d = dict()
  for i  in range(len(names)) : 
    d[names[i]]= [importances[i], importances1[i]]

 
  data = pd.DataFrame(d).T
  data.columns = ['RF', 'DT']
  data = data.sort_values('RF')

  # Plot the feature importance plot
  data.plot.barh(figsize=(15,9))
  plt.xlabel('Importance',fontsize=16)
  plt.title('Feature Importance Plot',fontsize=16)
  plt.show()
  
  # Most imporatant features 
  models=[model,model1]
  pair = [indices[0],indices[1]]
  
  plt.figure(figsize=(20,7))
  for k in range(2) :
    plt.subplot(1,2,k+1)
    # We only take the two corresponding features
    Xpair = X_train.iloc[:,  pair].reset_index(drop = True)
    ypair = y_train.reset_index(drop = True)
    
    # Fit the model 
    clf = models[k].fit(Xpair, ypair)

    x_min, x_max = Xpair.iloc[:, 0].min() - 1, Xpair.iloc[:, 0].max() + 1
    y_min, y_max = Xpair.iloc[:, 1].min() - 1, Xpair.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    
    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    
    # Plot the training points
    j = 0
    for i, color in zip(list(range(2)), "ym"):
        idx = np.where(ypair == i)[0]
        plt.scatter(Xpair.iloc[idx, 0], Xpair.iloc[idx, 1], c=color, label=j,
                    edgecolor='black', s=15)
        j+=1

    plt.title('Decision surface of  ' + data.columns[k] + '  using couples of the most important features')
  
  plt.show()













