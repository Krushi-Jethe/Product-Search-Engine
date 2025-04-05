# pylint: disable=import-error

"""
Docstring
"""

import io
import base64
from PIL import Image
from flask import Flask, request, render_template
from src.product_search_engine import ProductSearchEngine

app = Flask(__name__)
search_engine = ProductSearchEngine()


@app.route("/")
def home():
    """
    Docstring

    Returns:
        _type_: _description_
    """
    return render_template("index.html")


@app.route("/text_search", methods=["POST"])
def text_search():
    """
    Docstring

    Returns:
        _type_: _description_
    """

    inp_text = str(request.form["input_text"])
    search_results = search_engine.text_search(inp_text=inp_text)

    if isinstance(search_results, str):
        return render_template("index.html", prediction_text=search_results)

    data_urls = gen_data_urls(search_results=search_results)
    return render_template("index.html", data_urls=data_urls)


@app.route("/visual_search", methods=["POST"])
def visual_search():
    """
    Docstring

    Returns:
        _type_: _description_
    """

    inp_img = request.files["image_file"]
    inp_img = Image.open(io.BytesIO(inp_img.read()))

    search_results = search_engine.visual_search(inp_img=inp_img)

    if isinstance(search_results, str):
        return render_template("index.html", prediction_text=search_results)

    # Convert the list of PIL images to a list of data URLs
    data_urls = gen_data_urls(search_results=search_results)
    return render_template("index.html", data_urls=data_urls)


@app.route("/audio_search", methods=["POST"])
def audio_search():
    """
    Docstring

    Returns:
        _type_: _description_
    """

    search_results = search_engine.audio_search()

    if isinstance(search_results, str):
        return render_template("index.html", prediction_text=search_results)

    data_urls = gen_data_urls(search_results=search_results)
    return render_template("index.html", data_urls=data_urls)


def gen_data_urls(search_results):
    """
    Convert the list of PIL images to a list of data URLs

    Args:
        search_results (_type_): _description_

    Returns:
        _type_: _description_
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
    app.run(debug=True)
