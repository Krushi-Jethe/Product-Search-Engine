"""
Docstring
"""

import numpy as np
import speech_recognition as sr
from sklearn.metrics.pairwise import cosine_similarity
from data_and_models import (
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
    Docstring
    """

    def __init__(self):
        self.data = dataset
        self.knn = knn
        self.text_encoder = text_encoder
        self.vit_processor = vit_processor
        self.vit_model = vit_model
        self.vilt_processor = vilt_processor
        self.vilt_model = vilt_model

    def text_search(self, inp_text: str):
        """
        Docstring

        Args:
            inp_text (str): _description_

        Returns:
            _type_: _description_
        """

        word_embedding = self.text_encoder.encode(inp_text.lower())

        search_class = []
        for product in distinct_products:
            category = product.split()
            input_split = inp_text.split()
            for inputs in input_split:
                if inputs.lower() in category:
                    search_class.append(category)
        search_class = [" ".join(sublist) for sublist in search_class]

        if len(search_class) == 0:
            return "Sorry, we could not find what you are looking for!"

        filtered_df = self.data[self.data["articleType"].isin(search_class)]
        embeddings = self.text_encoder.encode(
            filtered_df["productDisplayName"].tolist()
        )
        cosine_similarities = cosine_similarity(
            word_embedding.reshape(1, -1), embeddings
        )
        indices = np.argsort(cosine_similarities[0])[::-1][:12]  # n=12
        similar_products = filtered_df.iloc[indices]
        return [similar_products.iloc[i]["image"] for i in range(12)]

    def audio_search(self):
        """
        Docstring

        Returns:
            _type_: _description_
        """

        inp_audio = ProductSearchEngine._audio_to_text()
        products = self.text_search(inp_audio)
        return products

    def visual_search(self, inp_img: np.ndarray):
        """
        Docstring

        Args:
            inp_img (np.ndarray): _description_

        Returns:
            _type_: _description_
        """

        inp_img.to(device)

        scores = {}
        for product in distinct_products:
            encoding = self.vilt_processor(inp_img, product, return_tensors="pt")
            outputs = self.vilt_model(**encoding)
            scores[product] = outputs.logits[0, :].item()

        is_present = any(score > 4 for score in scores.values())

        if is_present:
            inputs = self.vit_processor(inp_img, return_tensors="pt")
            output = self.vit_model(**inputs)
            output_embedding = output.logits.detach().numpy()
            _, indices = self.knn.kneighbors(output_embedding)
            indices = list(indices[0])

            return [self.data["train"]["image"][i] for i in indices]

        return "Sorry, we could not find what you are looking for!"

    @staticmethod
    def _audio_to_text():
        """
        Docstring
        """

        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Listening... Please speak clearly.")

            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                return text

            except sr.WaitTimeoutError:
                return "Listening timed out while waiting for phrase to start."

            except sr.UnknownValueError:
                return "Could not understand the audio."

            except sr.RequestError as e:
                return f"Could not request results from Google Speech Recognition service; {e}"
