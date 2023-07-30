"""


"""

from flask import Flask, request, render_template
from text_search import Text_search 
from visual_search import enhanced_Visual_search
from PIL import Image
import io
import base64

app = Flask(__name__)

Text_obj = Text_search()
Visual_obj = enhanced_Visual_search()

@app.route('/')
def home():
    return render_template('index.html')
  
@app.route('/text_search',methods=['POST'])
def text_search():
    
    inp_text = str(request.form['input_text'])
    prediction_text = Text_obj.run(input_text=inp_text)
    
    if prediction_text == 'Sorry, we could not find what you are looking for!':
        
            return render_template('index.html', prediction_text = prediction_text)
    else:
        
            # Convert the list of PIL images to a list of data URLs
            data_urls = []
            for encoding in prediction_text:
                buffered = io.BytesIO()
                encoding.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                data_url = f"data:image/jpeg;base64,{img_str}"
                data_urls.append(data_url)
            return render_template('index.html', data_urls = data_urls)
        
        
@app.route('/visual_search',methods=['POST'])
def visual_search():
    
    inp_img = request.files['image_file']
    inp_img = Image.open(io.BytesIO(inp_img.read()))
    
    prediction_image = Visual_obj.run(inp_img)
    
    if prediction_image == 'Sorry, we could not find what you are looking for!':
        
            return render_template('index.html', prediction_text = prediction_image)
    else:
        
            # Convert the list of PIL images to a list of data URLs
            data_urls = []
            for encoding in prediction_image:
                buffered = io.BytesIO()
                encoding.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                data_url = f"data:image/jpeg;base64,{img_str}"
                data_urls.append(data_url)
            return render_template('index.html', data_urls = data_urls)
           


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
