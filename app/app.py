# pylint: disable=import-error, wrong-import-position

"""
Flask app to run product search engine.
"""

import base64
import io
import os
import sys
from typing import List

from PIL import Image
from flask import Flask, request, render_template

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.product_search_engine import ProductSearchEngine



app = Flask(__name__)
search_engine = ProductSearchEngine()


@app.route("/")
def home():
    """
    Displays home page.

    Returns:
        str: Rendered HTML of the home page.
    """
    return render_template("index.html")


@app.route("/text_search", methods=["POST"])
def text_search() -> str:
    """
    Handles the text-based product search request.

    Returns:
        str: Rendered HTML of the results page.
    """

    inp_text = str(request.form["input_text"])
    search_results = search_engine.text_search(inp_text=inp_text)

    if isinstance(search_results, str):
        return render_template("index.html", message=search_results)

    data_urls = gen_data_urls(search_results=search_results)
    return render_template("index.html", data_urls=data_urls)


@app.route("/visual_search", methods=["POST"])
def visual_search() -> str:
    """
    Handles the image-based product search request.

    Returns:
        str: Rendered HTML of the results page.
    """

    inp_img = request.files["image_file"]
    inp_img = Image.open(io.BytesIO(inp_img.read())).convert("RGB")

    search_results = search_engine.visual_search(inp_img=inp_img)

    if isinstance(search_results, str):
        return render_template("index.html", message=search_results)

    # Convert the list of PIL images to a list of data URLs
    data_urls = gen_data_urls(search_results=search_results)
    return render_template("index.html", data_urls=data_urls)


@app.route("/audio_search", methods=["POST"])
def audio_search() -> str:
    """
    Handles the audio-based product search request.

    Returns:
        str: Rendered HTML of the results page.
    """

    search_results = search_engine.audio_search()

    if isinstance(search_results, str):
        return render_template("index.html", message=search_results)

    data_urls = gen_data_urls(search_results=search_results)
    return render_template("index.html", data_urls=data_urls)


def gen_data_urls(search_results: List[Image.Image]) -> List[str]:
    """
    Convert a list of PIL Image objects to a list of base64-encoded data URLs.

    Args:
        search_results (List[Image.Image]): A list of PIL Image objects representing search results.

    Returns:
        List[str]: A list of base64-encoded image data URLs to be rendered in HTML.
    """

    data_urls = []
    for encoding in search_results:
        buffered = io.BytesIO()
        encoding.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{img_str}"
        data_urls.append(data_url)

    return data_urls


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
