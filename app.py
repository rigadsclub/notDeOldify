import os
import sys
import requests
import ssl

from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file


from app_utils import bootstrap, upload_directory
from app_utils import results_img_directory, download
from app_utils import generate_random_filename, clean_me
from app_utils import clean_all, create_directory, get_model_bin
from app_utils import convertToJPG, allowed_extensions
from app_utils import allowed_file

from os import path

import torch
import fastai

from notdeoldify.lib import get_image_colorizer

import traceback

app = Flask(__name__)

# define a predict function as an endpoint
@app.route("/process", methods=["POST"])
def process_image():

    input_path = generate_random_filename(upload_directory,"jpeg")
    output_path = os.path.join(results_img_directory, os.path.basename(input_path))

    print(request.files)
    try:
        if 'file' in request.files:
            file = request.files['file']
            if allowed_file(file.filename):
                file.save(input_path)
            try:
                render_factor = request.form.getlist('render_factor')[0]
            except:
                render_factor = 30
            
        else:
            url = request.json["url"]
            download(url, input_path)

            try:
                render_factor = request.json["render_factor"]
            except:
                render_factor = 30

        try:
            image_colorizer.plot_transformed_image(path=input_path, out_path = output_path, figsize=(20,20),
                render_factor=int(render_factor), display_render_factor=True, compare=False)
        except:
            convertToJPG(input_path)
            image_colorizer.plot_transformed_image(path=input_path, out_path = output_path, figsize=(20,20),
            render_factor=int(render_factor), display_render_factor=True, compare=False)

        callback = send_file(output_path, mimetype='image/jpeg')
        
        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        pass
        clean_all([
            input_path,
            output_path
            ])


if __name__ == '__main__':
    bootstrap()
    global image_colorizer 
    image_colorizer = get_image_colorizer()

    port = 5000
    host = "0.0.0.0"
    app.run(host=host, port=port, threaded=False)
