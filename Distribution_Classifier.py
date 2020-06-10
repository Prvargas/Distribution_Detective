import re
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import *
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm 
import seaborn as sns
import pylab as py 
warnings.filterwarnings('ignore')


#Function that determines if a column is Continous or Discrete
def Dist_Type(df, col):
    dist_type = ''
    
    if df[col].dtype==float:
        dist_type='Continuous'
    else:
        dist_type='Discrete'
        
    return dist_type


def standarise(df, column,pct,pct_lower):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    sc = StandardScaler() 
    y = df[column][df[column].notnull()].to_list()
    y.sort()
    len_y = len(y)
    y = y[int(pct_lower * len_y):int(len_y * pct)]
    len_y = len(y)
    yy=([[x] for x in y])
    sc.fit(yy)
    y_std =sc.transform(yy)
    y_std = y_std.flatten()
    return y_std,len_y,y


#Function that applies chi square test for 5 dist classes in project
def Good_Fit(df, col):
    from scipy.stats import chisquare
    import numpy as np
    
    chi_dict = {}
    
    #BERNOULLI
    #Calc probability
    p = (df[col].value_counts()/len(df[col])).max()
    
    #Create bernoulli array
    bern_array = np.random.binomial(1, p, size=len(df))
    
    #Add bern results to chi square dict
    chi_dict['bernoulli'] = chisquare(f_obs=df[col]+1,f_exp= bern_array+1)
    
    
    #POISSON
    #Calc probability
    l = df[col].mean()
    #Create poisson array
    poi_array = np.random.poisson(lam=l, size=len(df))
    #Add poisson results to chi square dict
    chi_dict['poisson'] = chisquare(f_obs=np.histogram(df[col])[0]+1,f_exp= np.histogram(poi_array)[0]+1)
    
    
    
    #UNIFORM
    a=df[col].min()
    b=df[col].max()
    #Create uniform array
    uni_array = np.random.uniform(low=a, high=b, size=len(df))
    #Add uniform results to chi square dict
    chi_dict['uniform'] = chisquare(f_obs=np.histogram(df[col])[0],f_exp= np.histogram(uni_array)[0])
    
    
    
    #EXPONENTIAL
    l = df[col].mean()
    #Create exponential array
    exp_array = np.random.exponential(scale=l, size=len(df))
    #Add exponential results to chi square dict
    chi_dict['expon'] = chisquare(f_obs=np.histogram(df[col])[0]+1,f_exp= np.histogram(exp_array)[0]+1)
    
    
    #NORMAL (GAUSSIAN)
    mu = df[col].mean()
    std = df[col].std()
    #Create normal array
    norm_array = np.random.normal(loc=mu, scale=std, size=len(df))
    #Add normal results to chi square dict
    chi_dict['norm'] = chisquare(f_obs=np.histogram(df[col])[0],f_exp= np.histogram(norm_array)[0])

    
    
    idx= ['stat', 'pvalue']
    
    results_df = pd.DataFrame(chi_dict, index=idx)
    
    results_df = results_df.T.sort_values(by='stat')
    
    results_df.drop(columns='pvalue', inplace=True)
    
    return round(results_df)


# +
#Fit dist to all contiuous dist

def fit_distribution(df, column, pct=.99 ,pct_lower=.01):
    import numpy as np
    import re
    import pandas as pd
    import numpy as np
    import scipy.stats
    from sklearn.preprocessing import StandardScaler
    import math
    import matplotlib.pyplot as plt
    import warnings
    import statsmodels.api as sm 
    import seaborn as sns
    import pylab as py 
    warnings.filterwarnings('ignore')

    
    y_std,size,y_org = standarise(df, column,pct,pct_lower)
    '''dist_names = ['weibull_min','norm','weibull_max','beta',
                 'invgauss','uniform','gamma','expon', 'lognorm','pearson3','triang']'''
    
    dist_names = ['expon','norm', 'uniform']
    

    chi_square_statistics = []
    # 11 bins
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions
    
    param_dict = {}
    
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        #print("{}\n{}\n".format(dist, param))
        param_dict[distribution] = param

        # Get expected counts in percentile bins
        # cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = round(sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency),0)
        chi_square_statistics.append(ss)
    
    #Add params to a dataframe
    df_param = pd.DataFrame()

    for k, v in param_dict.items():
        df2 = pd.DataFrame({k:v})
        df_param = pd.concat([df_param,df2], ignore_index=False, axis=1)


    #Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True)


    print ('\nDistributions listed by Betterment of fit:')
    print ('............................................')
    print (results)
    return df_param


# +
#function that will train new CNN model

def CNN_Categorical_Model(train_dir = 'train_pics/', val_dir = 'val_pics/', classes=5, epochs=2 ):
    
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras import backend as K


    # dimensions of our images.
    img_width, img_height = 150, 150

    train_data_dir = train_dir
    validation_data_dir = val_dir
    nb_train_samples = 5000
    nb_validation_samples = 1000
    epochs = epochs
    batch_size = 500

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    print(train_generator.class_indices)
    
    class_label = train_generator.class_indices
    
    return model, class_label


# +
#This classifies a column based on a saved h5 model

def Image_Classifier(df, col, h5 ='hist_classifier_v2' ):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from keras.models import load_model

    #Create Hist of desired column and save img
    df[col].hist()
    plt.savefig('hist_img_test.png');
    
    # load model
    model = load_model(h5)
    #print(model.summary())
    
    #Reload the saved image and used the model to predict dist type
    image = tf.keras.preprocessing.image.load_img('hist_img_test.png', target_size=(150,150))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    yFit = model.predict_classes(input_arr)
    
    print('CNN Model\nPrediction Index: {}\n'.format(yFit))
    
    #post which use the train generator to map the labels 
    #back to actual names of the classes 
    
    #This is the dict generated by the train_generator
    labels = {'bernoulli': 0, 'expon': 1, 'norm': 2, 'poisson': 3, 'uniform': 4}

    label_map = dict((v,k) for k,v in labels.items()) #flip k,v
    pred_class = [label_map[k] for k in yFit]

    
    print('Prediction class: {}'.format(pred_class))
    
    return pred_class



# +
#Write a function that will convert a df column to hist image array for keras predictions

def Col_To_Img_Array(df, col):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from keras.models import load_model
    
    #Create Hist of desired column and save img
    df[col].hist()
    plt.savefig('Col_To_Array.png');
    
    #Reload the saved image and used the model to predict dist type
    image = tf.keras.preprocessing.image.load_img('Col_To_Array.png', target_size=(150,150))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    
    return input_arr


# -

#This function randomly generate normal dist and saves the imgs
def Norm_Img_Gen(n=2, filepath = 'train_pics/norm/'):
    import matplotlib.pyplot as plt

    for i in range(n):
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        bins = np.random.randint(5,50)
        mean = np.random.uniform(1,200)
        std = np.random.uniform(1,10)
        print('mean:{:.02f}  std:{:.02f}  bins:{}'.format(mean, std, bins))
        y_ = np.random.normal(mean, std, 5000)
        ax.hist(y_, bins=bins)
        fig.savefig('{}norm_mean-{:.02f}_std-{:.02f}_bins-{}.png'.format(filepath, mean, std, bins))
        plt.close(fig);


# +
#This function randomly generate bernoulli dist and saves the imgs

def Bern_Img_Gen(n=2, filepath = 'train_pics/bernoulli/' ):
    import matplotlib.pyplot as plt

    for i in range(n):
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        bins = np.random.randint(5,50)
        
        n = 1 #np.random.randint(1,200)
        p = np.random.uniform(0,1)
        print('n:{}  p:{:.02f}  bins:{}'.format(n, p, bins))
        y_ = np.random.binomial(n, p, 5000)
        ax.hist(y_, bins=bins)
        fig.savefig('{}bern_n-{}_p-{:.04f}_bins-{}.png'.format(filepath, n, p, bins))
        plt.close(fig);


# +
#This function randomly generate exponential dist and saves the imgs

def Exp_Img_Gen(n=2, filepath = 'train_pics/expon/'):
    import matplotlib.pyplot as plt
    
    for i in range(n):
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        bins = np.random.randint(5,50)
        
        lam = np.random.uniform(1,20)
        print('lambda:{:.02f}  bins:{}'.format(lam, bins))
        y_exp = np.random.exponential(lam, 5000)
        ax.hist(y_exp, bins=bins)
        fig.savefig('{}exp_lam-{:.04f}_bins-{}.png'.format(filepath, lam, bins))
        plt.close(fig);


# +
#This function randomly generate poisson dist and saves the imgs

def Poi_Img_Gen(n=2, filepath = 'train_pics/poisson/' ):
    import matplotlib.pyplot as plt
    
    for i in range(n):
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        bins = np.random.randint(5,50)
        
        
        lam = np.random.uniform(1,20)
        print('lambda:{:.02f}  bins:{}'.format(lam, bins))
        y_poi = np.random.poisson(lam, 5000)
        ax.hist(y_poi, bins=bins)
        fig.savefig('{}poisson_lam-{:.04f}_bins-{}.png'.format(filepath, lam, bins))
        plt.close(fig);


# +
#This function randomly generate uniform dist and saves the imgs

def Uni_Img_Gen(n=2, filepath = 'train_pics/uniform/' ):
    import matplotlib.pyplot as plt
    
    for i in range(n):
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        bins = np.random.randint(5,50)
        
        a = np.random.uniform(1,200)
        b = a + np.random.uniform(1,20)
        print('a:{:.02f} b:{:.02f}  bins:{}'.format(a, b, bins))
        y_ = np.random.uniform(a, b, 5000)
        ax.hist(y_, bins=bins)
        fig.savefig('{}uniform_a-{:.04f}_b-{:.04f}_bins-{}.png'.format(filepath, a, b, bins))
        plt.close(fig);


# -

def Distribution_Detective(df, col):
    
    Type = Dist_Type(df, col)
    
    print(Type,'\n\n')
    
    fit_df = Good_Fit(df, col)
    
    print(fit_df, '\n\n')
    
    pred = Image_Classifier(df, col, h5 ='hist_classifier_v2' )
    
    return Type, fit_df, pred


