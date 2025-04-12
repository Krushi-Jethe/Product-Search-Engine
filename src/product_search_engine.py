"""
class: ProductSearchEngine

Performs product search based on text, audio, or image input.
- Semantic search using text input.
- Audio-based search via speech-to-text conversion.
- Visual search using image input and multi-modal models.
"""

from typing import Tuple, Union
import torch
from PIL import Image
import numpy as np
import speech_recognition as sr
from sklearn.metrics.pairwise import cosine_similarity
from .data_and_models import (
    device,
    dataset,
    knn,
    text_encoder,
    vit_processor,
    vit_model,
    vilt_processor,
    vilt_model,
)

distinct_products = set(dataset["articleType"])


class ProductSearchEngine:
    """
    --------------------------
    Attributes:
    --------------------------
    1. data - Dataframe consisting of images, product_category, product_description
    2. knn - KNN fit on image embeddings
    3. text_encoder - to encode text and generate embeddings
    4. vit_processor - to process image before passing to vit_model
    5. vit_model - to generate embeddings of images.
    6. vilt_processor - to process image before passing to vilt_model
    7. vilt_model - to classify input image for classes present in dataset

    --------------------------
    Methods:
    --------------------------
    1. text_search - performs text based search for products
    2. audio_search - performs audio based search for products
    3. visual_search - performs image based search for products
    4. _audio_to_text - records audio from microphone and converts
                        to text
    """

    def __init__(self):
        self.data = dataset
        self.knn = knn
        self.text_encoder = text_encoder
        self.vit_processor = vit_processor
        self.vit_model = vit_model
        self.vilt_processor = vilt_processor
        self.vilt_model = vilt_model

    @torch.no_grad
    def text_search(self, inp_text: str) -> Union[list, str]:
        """
        Performs semantic search by computing cosine similarity between
        the input text and product descriptions in the dataset.

        Args:
            inp_text (str): User input text for performing the search.

        Returns:
            Union[list, str]: list of similar images if present or a message.
        """

        word_embedding = self.text_encoder.encode(inp_text.lower())

        embeddings = self.text_encoder.encode(
            (self.data["productDisplayName"] + " " + self.data["articleType"]).tolist()
        )

        cosine_similarities = cosine_similarity(
            word_embedding.reshape(1, -1), embeddings
        )

        indices = np.argsort(cosine_similarities[0])[::-1][:12]  # n=12
        threshold = 0.4
        filtered_indices = [i for i in indices if cosine_similarities[0][i] > threshold]

        if not filtered_indices:
            return "Sorry, we could not find what you are looking for!"
        
        similar_products = self.data.iloc[indices]
        return [similar_products.iloc[i]["image"] for i in range(12)]

    def audio_search(self) -> Union[list, str]:
        """
        Listens to audio input using microphone, converts it to text and
        then performs text search.

        Returns:
            Union[list, str]: list of similar images if present or a message.
        """

        inp_audio, flag = ProductSearchEngine._audio_to_text()
        if flag != "success":
            return inp_audio
        products = self.text_search(inp_audio)
        return products

    @torch.no_grad
    def visual_search(self, inp_img: Image.Image) -> Union[list, str]:
        """
        Checks if image class is present in the dataset using Vision-Language model
        for classification and then fetches similar images.

        Args:
            inp_img (PIL.Image.Image): User input image for performing the search.

        Returns:
            Union[list, str]: list of similar images if present or a message.
        """

        inp_img = torch.from_numpy(np.array(inp_img)).float()
        inp_img.to(device)

        scores = {}
        for product in distinct_products:
            encoding = self.vilt_processor(inp_img, product, return_tensors="pt")
            encoding.to(device)
            outputs = self.vilt_model(**encoding)
            scores[product] = outputs.logits[0, :].item()
            print(product, scores[product])

        is_present = any(score > 4 for score in scores.values())

        if is_present:
            inputs = self.vit_processor(inp_img, return_tensors="pt")
            inputs.to(device)
            output = self.vit_model(**inputs)
            output_embedding = output.logits.detach().cpu().numpy()
            _, indices = self.knn.kneighbors(output_embedding)
            indices = list(indices[0])

            return [self.data["image"][i] for i in indices]

        return "Sorry, we could not find what you are looking for!"

    @staticmethod
    def _audio_to_text() -> Tuple[str, str]:
        """
        Uses speech_recognition to record audio from
        device microphone and convert it to text.

        Returns:
            Tuple[str, str]: recorded audio, status (successful or unsuccessful)
        """

        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Listening... Please speak clearly.")

            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10)
                text = recognizer.recognize_google(audio)
                return text, "success"

            except sr.WaitTimeoutError:
                return (
                    "Listening timed out while waiting for phrase to start.",
                    "unsuccessful",
                )

            except sr.UnknownValueError:
                return "Could not understand the audio.", "unsuccessful"

            except sr.RequestError as e:
                return (
                    f"Could not request results from Google Speech Recognition service; {e}",
                    "unsuccessful",
                )
