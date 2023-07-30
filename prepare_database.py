import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import cv2

warnings.filterwarnings("ignore")


class create_data():
    
    """
        create_data class is designed to preprocess and filter a dataset to obtain balanced data for training a neural network.
        
        Attributes:
            path (str): The file path to the CSV file containing the original dataset.
        
        Methods:
            run():
                Performs the preprocessing and filtering of the dataset to obtain balanced data for training.
                It prints information about the dataset, unique article types, article type counts,
                considers articles with at least 300 entries, gets rid of articles with <300 entries,
                samples 300 articles from each articleType to achieve balanced data for training.
    """

    def __init__(self,path):
        
        """
        Initializes the create_data class by reading the CSV file containing the original dataset.

        Parameters:
            path (str): The file path to the CSV file containing the original dataset.
        """
        self.df = pd.read_csv(path)
        self.df.drop(['Unnamed: 10','Unnamed: 11','gender','masterCategory','subCategory','baseColour','season','year','usage'],axis=1,inplace=True) #,'productDisplayName'
    
    def run(self):
        
        """
        Performs the preprocessing and filtering of the dataset to obtain balanced data for training.
        It prints information about the dataset, unique article types, article type counts,
        considers articles with at least 300 entries, gets rid of articles with <300 entries,
        samples 300 articles from each articleType to achieve balanced data for training.
        """
        
        print('*******************************************************************')
        print('info of the data')
        print('*******************************************************************\n\n')
        self.df.info()
        
        print('\n\n*******************************************************************')
        print('It contains unique article types as given below')
        print('*******************************************************************\n\n')
        print(self.df['articleType'].unique())
        
        print('\n\n*******************************************************************')
        print('article type counts - printing only 50')
        print('*******************************************************************\n\n')
        article_type_counts = self.df['articleType'].value_counts()
        print(article_type_counts.head(50))
        
        print('\n\n*******************************************************************')
        print('Consider articles that contain atleast 300 entries')
        print('*******************************************************************\n\n')
        filter_articles = article_type_counts[article_type_counts>=300]
        print(filter_articles)
        
        print('\n\n*******************************************************************')
        print('Getting rid of articles having <300 entries')
        print('*******************************************************************\n\n')
        filtered_article_types = article_type_counts[article_type_counts > 300].index
        filtered_data = self.df[self.df['articleType'].isin(filtered_article_types)]
        filtered_data.info()
        
        print('\n\n*******************************************************************')
        print('Sampling 300 articles from each articleType so that data is balanced for training the network')
        print('*******************************************************************\n\n')
        label_encoder = LabelEncoder()
        filtered_data['articleType_label'] = label_encoder.fit_transform(filtered_data['articleType'])

        grouped = filtered_data.groupby('articleType_label')
        filtered_data = grouped.apply(lambda x: x.sample(300))
        filtered_data = filtered_data.reset_index(drop=True)

        self.final_data = filtered_data
        print('To access final data --- use final_data attribute of the object and to access original data use the df attribute of the object')
        
        
        
        
class final_imgs():
    
    """
    final_imgs class is designed to resize and save images from a dataset into separate directories
    based on their respective articleType labels.

    Attributes:
        df (pandas.DataFrame): The DataFrame containing the dataset with articleType labels.
        save_img_dir (str): The directory path where resized images will be saved.
        original_img_dir (str): The directory path where original images are stored.

    Methods:
        run():
            Resizes and saves images into separate directories based on their articleType labels.
    """
    
    
    def __init__(self,df,save_img_dir,original_img_dir):
        
        """
        Initializes the final_imgs class.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the dataset with articleType labels.
            save_img_dir (str): The directory path where resized images will be saved.
            original_img_dir (str): The directory path where original images are stored.
        """
        
        self.df = df
        self.save_img_dir = save_img_dir
        self.original_img_dir = original_img_dir
        
    def run(self):
        
        """
        Resizes and saves images into separate directories based on their articleType labels.
        """
        
        image_paths = [self.original_img_dir + '/' + i for i in os.listdir(self.original_img_dir)]
        
        for i in range(31):
            os.mkdir(self.save_img_dir + str(i))
            
        dir_num = 0
        path = self.save_img_dir + str(dir_num)
        for i in range(len(self.df)):
            if (i)%300 == 0 and i!=0:
                dir_num+=1
                path = self.save_img_dir + str(dir_num)

            for j in image_paths:
                if j.split('/')[-1][:-4] == str(self.df['id'][i]):
                    img = cv2.imread(j)
                    img = cv2.resize(img,(256,256))
                    cv2.imwrite(path + '/' + str(self.df['id'][i])+'.jpg',img)    
                    self.df['id'][i] = str(self.df['id'][i]) + '.jpg'
        
        
        print('Process Completed.................')
        
        
        
        
def regroup_imgs(path_to_img_dir , path_to_metadata , save_to_dir):
    
        """
        regroup_imgs is used to store all the images and metadata.csv into a single folder
        path_to_img_dir : path where output of final_imgs.run was stored
        path_to_metadata : path where dataframe of final_imgs.run was stored
        save_to_dir : New data where the metadata + images will be regrouped
    
       """
    
        path_list = [path_to_img_dir+'/'+i+'/'+j for i in os.listdir(path_to_img_dir) for j in os.listdir(path_to_img_dir+'/'+i)]    
        df = pd.read_csv(path_to_metadata)
        df.to_csv(save_to_dir+'/metadata.csv')
        for i in range(len(path_list)):
            img = cv2.imread(path_list[i])
            cv2.imwrite(save_to_dir+'/'+path_list[i].split('/')[-1],img)
        
