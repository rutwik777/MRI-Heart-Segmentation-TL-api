import flask
from numpy.lib.type_check import imag
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image
import base64
import io

app = Flask(__name__)
device = torch.device('cpu')
model = torch.load('u-effb4_100.pth', map_location=torch.device('cpu'))
model.to(device)
model.eval()
data = io.BytesIO()
convert_tensor = transforms.ToTensor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results in HTML
    '''
    image = request.files["file"]
    #image = image.read()
    print('debug',image)
    #print('debug', image.type())
    #image = np.frombuffer(image)
    #image = image.reshape((1, 96, 96))
    #print('debug',image)

    #image = torch.from_numpy(image)
    #print('debug',image)
    image = Image.open(image)
    data = io.BytesIO()
    image.save(data, "PNG")
    image = convert_tensor(image)
    image = image.reshape((1, 96, 96))
    image = image.unsqueeze(1)
    print('debug',image)
    image_1 = image.to(device)
    output = model(image_1)
    print('debug', output)
    output = transforms.ToPILImage()(output.squeeze(0))
    print('debug', output)
    data = io.BytesIO()
    output.save(data, "PNG")
    print('debug', output)
    encoded_img_data = base64.b64encode(data.getvalue())
    #out_np = torch.max(output,1).indices.cpu().detach().numpy()
    #encoded_img_data = base64.b64encode(output.getvalue())
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = np.round(prediction, 0)
    #output = int(output)
    #output = "${:,.2f}".format(output)

    #return render_template('index.html', prediction_text='House price should be {}'.format(output))
    return render_template('index.html', prediction=encoded_img_data.decode('utf-8'))

if __name__ == "__main__":
    app.run(debug=True)