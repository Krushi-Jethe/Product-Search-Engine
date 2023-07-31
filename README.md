# Image-Search-Engine
An Image search engine which searches for images from a database. Takes either text, image or audio as input to give relevant output.

For the text based search, we have used BERT tokenizer to tokenize the words. Cosine similarity between input text and 'productDisplayname' is used to fetch relevant images.

In case of visual search where the user uploads an image, 
--First the uploaded image is passed through ViLT model with pre-defined text queries(classes present in the dataset for e.g. Backpacks, Belts, Deodarants etc.) 
--This ViLT model scores the image for all the 31 classes/the text queries. 
--If the score is >4 then the input image is passed through ViT to extract embeddings and KNN is performed to retrieve the nearest neighbours of this image.

In audio based search , the audio is first converted to text and then text based search is performed.


![Image Search engine](https://github.com/Krushi-Jethe/Image-Search-Engine/assets/137395922/11ac020c-cb13-48f9-a267-d064b5e9a3f7)

Further improvements will also be added using InstructBLIP/any other similar model for image descriptions + reinforcement learning to improve results. And also increase our database by adding the user uploaded images.

Database -- https://huggingface.co/datasets/KrushiJethe/fashion_data
