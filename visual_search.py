from datasets import load_dataset
import numpy as np
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification 
from sklearn.neighbors import NearestNeighbors
from transformers import ViltProcessor, ViltForImageAndTextRetrieval


class enhanced_Visual_search():
    
    """
    enhanced_Visual_search class is designed to perform enhanced visual search and retrieve similar images based on 
    input images and pre-defined text queries(self.text).

    Attributes:
        path_to_embeddings (str): The file path to the pre-computed image embeddings (default is the path
                                  'C:/Users/DELL/OneDrive/Documents/Python-Scripts/image_embeddings.npy').
        nearest_neighbours (int): The number of nearest neighbors to retrieve for each input image (default is 12).
        
    Methods:
        run(image):
            Performs the enhanced visual search and returns a list of similar images based on the input image and text queries.

    Example Usage:
    -------------
    # Initialize the enhanced_Visual_search class
    visual_search = enhanced_Visual_search()

    # Load an input image 
    input_image = an image

    # Perform the enhanced visual search and retrieve similar images
    similar_images = visual_search.run(input_image)

    """
    
    def __init__(self,path_to_embeddings='C:/Users/DELL/OneDrive/Documents/Python-Scripts/image_embeddings.npy',nearest_neighbours=12):
        
        """
        Initializes the enhanced_Visual_search class by loading pre-trained models, image embeddings, and fashion dataset.

        Parameters:
            path_to_embeddings (str): The file path to the pre-computed image embeddings (default is the path
                                      'C:/Users/DELL/OneDrive/Documents/Python-Scripts/image_embeddings.npy').
            nearest_neighbours (int): The number of nearest neighbors to retrieve for each input image (default is 12).
        """
        
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model_vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.extracted_embeddings = np.load(path_to_embeddings)
        self.knn = NearestNeighbors(n_neighbors = nearest_neighbours)
        self.knn.fit(self.extracted_embeddings)
        self.data = load_dataset('KrushiJethe/fashion_data')
        self.images = self.data['train']['image']
        self.titles = self.data['train']['productDisplayName']
        
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        self.vilt_model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        
        self.texts = ['Backpacks', 'Belts', 'Bra', 'Briefs', 'Casual Shoes', 'Deodorant', 'Dresses', 'Earrings', 'Flats', 'Flip Flops', 'Formal Shoes', 'Handbags', 'Heels', 'Jeans', 'Kurtas', 'Lipstick', 'Nail Polish', 'Perfume and Body Mist', 'Sandals', 'Sarees', 'Shirts', 'Shorts', 'Socks', 'Sports Shoes', 'Sunglasses', 'Tops', 'Track Pants', 'Trousers', 'Tshirts', 'Wallets', 'Watches']  #These are all the classes present in our dataset

        
        
    def run(self,image):
        
        """
        Performs visual search and returns a list of similar images based on the input image and text queries.
        The text queries are used to filter out inputs which are not present in our database.

        Parameters:
            image : The input image for which similar images need to be retrieved.

        Returns:
            list: A list containing similar images to the input image based on visual and text-based similarity.
                  Each element of the list is a PIL image to a similar image.
                  If no similar images are found, it returns the message 'Sorry, we could not find what you are looking for!'
        """

        # forward pass
        scores = dict()
        
        for text in self.texts:
            # prepare inputs
            encoding = self.vilt_processor(image, text, return_tensors="pt")
            outputs = self.vilt_model(**encoding)
            scores[text] = outputs.logits[0, :].item()


        temp=False
        for i in scores.values():
            if i>4:
                temp=True

        if temp==True:
                inputs = self.feature_extractor(image, return_tensors='pt') 
                output = self.model_vit(**inputs)
                temp = output.logits.detach().numpy()
                _ , indices = self.knn.kneighbors(temp)
                indices = list(indices[0])

                return [self.images[i] for i in indices]
    
        elif temp==False:
                return 'Sorry, we could not find what you are looking for!'

