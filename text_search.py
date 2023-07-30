from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Text_search():
    
    """
    Text_search class is designed to perform a text-based product search using pre-trained BERT-based sentence embeddings.

    Attributes:
        data: A dictionary containing the fashion dataset.
        df: A pandas DataFrame created from the 'train' subset of the fashion dataset.
        model_text: An instance of the SentenceTransformer model with 'bert-base-nli-mean-tokens' embedding.
    
    Methods:
        run(input_text, n=12):
            Performs the text-based product search using the input text and returns a list of similar product images.

    Example Usage:
    -------------
    text_search = Text_search()
    input_query = "blue dress for summer"
    similar_product_images = text_search.run(input_query)

    """
    
    def __init__(self):
        
        """
        Initializes the Text_search class by loading the fashion dataset, creating a DataFrame, and setting up the 
        SentenceTransformer model for text embedding.
        """
        
        self.data = load_dataset('KrushiJethe/fashion_data')
        self.df = pd.DataFrame(self.data['train'])
        self.df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
        self.df['articleType'] = self.df['articleType'].str.lower()
        self.df['productDisplayName'] = self.df['productDisplayName'].str.lower()
        self.model_text = SentenceTransformer('bert-base-nli-mean-tokens')

    def run(self, input_text, n=12):
        
        """
        Performs text-based product search using the input text and returns a list of similar product images.

        Parameters:
            input_text (str): The input text query for product search.
            n (int): The number of similar product images to return (default is 12).

        Returns:
            list: A list containing 'n' similar product images based on the input text query.
                  Each element of the list is a PIL image of a similar product.
                  If no similar products are found, it returns the message 'Sorry, we could not find what you are looking for!'
        """
        

        # Generate sentence embeddings for input text
        word_embedding = self.model_text.encode(input_text.lower())
        

        # Search for matching classes
        search_class = []
        for classname in self.df['articleType'].unique():
            category = classname.split()
            input_split = input_text.split()
            for inputs in input_split:
                if inputs.lower() in category:
                    search_class.append(category)
        search_class = [' '.join(sublist) for sublist in search_class]
        
        
        if len(search_class) == 0:
            
            return 'Sorry, we could not find what you are looking for!'
        
        else:
            
            # Filter dataframe based on search class
            filtered_df = self.df[self.df['articleType'].isin(search_class)]
    
            
            # Generate sentence embeddings for product display names
            embeddings = self.model_text.encode(filtered_df['productDisplayName'].tolist())
            
            # Compute pairwise cosine similarities
            cosine_similarities = cosine_similarity(word_embedding.reshape(1,-1), embeddings)
            indices = np.argsort(cosine_similarities[0])[::-1][:n]
            similar_products = filtered_df.iloc[indices]
            
            return [similar_products.iloc[i]['image'] for i in range(12)]
    