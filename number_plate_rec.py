import requests, os, json
  
url = "https://app.nanonets.com/api/v2/ObjectDetection/Model/"
api_key = os.environ.get('NANONETS_API_KEY')

##
payload = "{\"categories\" : [\"number_plate\"], \"model_type\": \"ocr\"}"
headers = {'Content-Type': "application/json",}

response = requests.request("POST", url, headers=headers, auth=requests.auth.HTTPBasicAuth(api_key, ''), data=payload)
model_id = json.loads(response.text)["model_id"]

print("NEXT RUN: export NANONETS_MODEL_ID=" + model_id)
print("THEN RUN: python ./code/upload-training.py")




import os
import urllib
import json
import pandas as pd
import xml.etree.cElementTree as ET

file = './data/indian_number_plates.json'
df = pd.read_json(file, lines=True)

df.dropna(subset=['annotation'], inplace=True)

annot = df['annotation'].values
annot = [x[0] for x in annot]

urls = df['content'].values

for i,a in enumerate(annot):

	w = a['imageWidth']
	h = a['imageHeight']

	root = ET.Element('annotation')
	filename = ET.SubElement(root, 'filename').text = '{}.jpg'.format(i)
	size = ET.SubElement(root, 'size')
	ET.SubElement(size, "width").text = str(int(w))
	ET.SubElement(size, "height").text = str(int(h))
	obj = ET.SubElement(root, 'object')
	ET.SubElement(obj, 'name').text = a['label'][0]
	bndbox = ET.SubElement(obj, 'bndbox')
	ET.SubElement(bndbox, 'xmin').text = str(int(a['points'][0]['x'] * w))
	ET.SubElement(bndbox, 'ymin').text = str(int(a['points'][0]['y'] * h))
	ET.SubElement(bndbox, 'xmax').text = str(int(a['points'][1]['x'] * w))
	ET.SubElement(bndbox, 'ymax').text = str(int(a['points'][1]['y'] * h))

	tree = ET.ElementTree(root)

	if not os.path.exists('./annotations/xmls'):
		os.makedirs('./annotations/xmls')

	file = './annotations/xmls/{}.xml'.format(i)
	with open(file, 'wb') as f:	
		tree.write(f, encoding='utf-8')

	# if not os.path.exists('./images'):
	# 	os.makedirs('./images')

	# urllib.urlretrieve(urls[i], './images/{}.jpg'.format(i))



    
model_id = os.environ.get('NANONETS_MODEL_ID')
api_key = os.environ.get('NANONETS_API_KEY')

url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id

response = requests.request('GET', url, auth=requests.auth.HTTPBasicAuth(api_key,''))

state = json.loads(response.text)["state"]
status = json.loads(response.text)["status"]

if state != 5:
	print("The model isn't ready yet, its status is:", status)
	print("We will send you an email when the model is ready. If you are impatient, run this script again in 10 minutes to check.")
else:
	print("NEXT RUN: python ./code/prediction.py ./images/151.jpg")



model_id = os.environ.get('NANONETS_MODEL_ID')
api_key = os.environ.get('NANONETS_API_KEY')
image_path = sys.argv[1]

url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/LabelFile/'

data = {'file': open(image_path, 'rb'),    'modelId': ('', model_id)}

response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)

print(response.text)




model_id = os.environ.get('NANONETS_MODEL_ID')
api_key = os.environ.get('NANONETS_API_KEY')

url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/Train/'

querystring = {'modelId': model_id}

response = requests.request('POST', url, auth=requests.auth.HTTPBasicAuth(api_key, ''), params=querystring)

print(response.text)

print("\n\nNEXT RUN: python ./code/model-state.py")



pathToAnnotations = './annotations/json'
pathToImages = './images'
model_id = os.environ.get('NANONETS_MODEL_ID')
api_key = os.environ.get('NANONETS_API_KEY')

for root, dirs, files in os.walk(pathToAnnotations, topdown=False):
    for name in tqdm(files):
        annotation = open(os.path.join(root, name), "r")
        filePath = os.path.join(root, name)
        imageName, ext = name.split(".")
        if imageName == "":
            continue
        imagePath = os.path.join(pathToImages, imageName + '.jpg')
        jsonData = annotation.read()
        url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/UploadFile/'
        data = {'file' :open(imagePath, 'rb'),  'data' :('', '[{"filename":"' + imageName+".jpg" + '", "object": '+ jsonData+'}]'),   'modelId' :('', model_id)}       
        response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)
        if response.status_code > 250 or response.status_code<200:
            print(response.text), response.status_code

print("\n\n\nNEXT RUN: python ./code/train-model.py")



import os
import json
from xmljson import parker
from xml.etree.ElementTree import fromstring
import argparse
from tqdm import tqdm

def keep_keys(old_dict):
  new_dict = {}
  for key in old_dict:
    if key in ["object","segmented","size"]:
      new_dict[key]=old_dict[key]
  return new_dict


parser = argparse.ArgumentParser(description='Convert xml Annotations to json annotations')
parser.add_argument('--xml', type=str,  metavar='path/to/input/xml/', default='./annotations/xmls/', help='(default "annotations/xml/") path to xml annotations')
parser.add_argument('--json', type=str,  metavar='path/to/output/json/', default='./annotations/json/', help='(default "annotations/json/") path to out json annotations')

parser.print_help()
print("\n")

args = vars(parser.parse_args())
print(args)

input_directory = args["xml"]
output_directory = args["json"]

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

all_input_files = os.listdir(input_directory)
xml_input_files = [file for file in all_input_files if file.endswith(".xml")]
for xml_file in tqdm(xml_input_files):
  f = open(os.path.join(input_directory, xml_file),"rb")
  json_dict = keep_keys(parker.data(fromstring(f.read())))
  if not "object" in json_dict:
    json_dict = {"object":{}}
  json_output = json.dumps(json_dict["object"])
  f.close()

  f = open(os.path.join(output_directory, xml_file.replace(".xml",".json")),"w")
  if json_output[0]!='[':
    json_output = '['+json_output+']'
  f.write(json_output)
  f.close()
print("\n\n\n")
print("Completed Parsing")




