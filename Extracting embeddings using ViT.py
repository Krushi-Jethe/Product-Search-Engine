#######################################################
#################EXTRACTING EMBEDDINGS#################
#######################################################


# Loading the dataset
from datasets import load_dataset
dataset = load_dataset("KrushiJethe/fashion_data")

#Loading the feature extractor for ViT
from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

#Loading the pretrained ViT model
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

#Batch processing the data to extract embedding and save them as numpy array
import numpy as np

x=0
for i in range(146):
    if i!= 145:
        inputs = feature_extractor(dataset['train']['image'][x:x+64] , return_tensors='pt')
        output = model(**inputs)
        np.save(str(x),output.logits.detach().numpy())
        x+=64
    elif i==145:
        inputs = feature_extractor(dataset['train']['image'][9280:9300] , return_tensors='pt')
        output = model(**inputs)
        np.save('9280',output.logits.detach().numpy())


#Merging all the saved batch embeddings
import os

lst = os.listdir('./')
lst = [x for x in lst if x[-3:]=='npy']
x=64
for i in range(146):
    if i==0:
        embedding = np.load('0.npy')
    else:
        temp = np.load(str(x)+'.npy')
        embedding = np.append(embedding,temp,axis=0)
        x+=64

#Check the shape of the embedding --- should be 9300x1000
embedding.shape

#Save the merged 9300 image embeddings
np.save('image_embeddings',embedding)

### ----------OPTIONAL---------- ###

#Now save all the images together as a numpy array
img_array = np.array(dataset['train']['image'])
img_array.shape #Check the shape of the image array -- should have 9300 images
np.save('image_array',img_array)