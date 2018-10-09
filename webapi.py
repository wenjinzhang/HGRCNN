from flask import Flask, render_template, request, jsonify
import base64
import time
from PIL import Image
import json
from model import ConvColumn
from torchvision.transforms import *
import torch
import torch.nn as nn
app = Flask('__HGR__')

IMGS_Array =[]
# load config file
with open('./configs/config2.json') as data_file:
    config = json.load(data_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

def recongnize(imgs):
    print('iscall')
    # pre-op data
    data = []
    for img in imgs:
        img = transform(img)
        data.append(torch.unsqueeze(img, 0))

    input = torch.cat(data)
    input = input.permute(1, 0, 2, 3)
    input = [input]
    # create model
    model = ConvColumn(config['num_classes'])

    # multi GPU setting
    model = torch.nn.DataParallel(model).to(device)
    checkpoint = torch.load(config['checkpoint'])
    model.load_state_dict(checkpoint['state_dict'])

    # compute the model
    input = input.to(device)
    out = model(input)
    print(out.detach().cpu().numpy)


@app.route("/receive", methods=['GET', 'POST'])
def receive_img():
    print('call me')
    data = request.get_json()
    imgdata = base64.b64decode(data['imageBase64'])
    # save imgs
    file = open("images/1.png", 'wb')
    file.write(imgdata)
    file.close()
    imgdata=Image.open("images/1.png").convert('RGB')
    # insert images
    IMGS_Array.append(imgdata)
    print(IMGS_Array)
    result_data = {}
    if len(IMGS_Array) > 18:
        del IMGS_Array[0]
    else:
        result_data['result'] = 'fail'
        result_data['info'] = 'don\'t have enough numbers'
        return jsonify(result=result_data)

    # recognize
    recongnize(IMGS_Array)

    return jsonify(result=data)


@app.route("/test", methods=['GET', 'POST'])
def test():
    data = request.get_json()
    print(data['name'])
    return 'hello world'

@app.route("/")
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')






