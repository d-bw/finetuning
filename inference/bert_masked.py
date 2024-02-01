from transformers import BertTokenizer, BertForMaskedLM ,pipeline
from datasets import load_dataset
import re
import json
from tqdm.auto import tqdm

def parse_polygon_string(polygon_string):
    # 通过正则表达式提取坐标信息
    coordinates = re.findall(r'\((\d+),(\d+)\)', polygon_string)  
    return [f"{x},{y}" for x, y in coordinates]

def parse_polygon_int(polygon_string):
    # 通过正则表达式提取坐标信息
    coordinates = re.findall(r'\((\d+),(\d+)\)', polygon_string)
    int_coordinates=[]
    for x,y in coordinates:
      int_coordinates.append((int(x),int(y)))
    return int_coordinates


  
def convert_to_json(input_string):
    # 通过正则表达式提取每个多边形的信息
    polygon_matches = re.findall(r'polygon\d+:\(\(\d+,\d+\),\(\d+,\d+\),\(\d+,\d+\),\(\d+,\d+\)\)', input_string)

    # 构建 Polygons 部分
    polygons = [{"polygon{}".format(i + 1): parse_polygon_string(match) } for i, match in enumerate(polygon_matches)]
    
    
    # 构建 Predictions 部分
    prediction_matches = re.findall(r'->\(\(\d+,\d+\),\(\d+,\d+\),\(\d+,\d+\),\(\d+,\d+\)\)', input_string)
    
    predictions = [{"polygon{}".format(i + 1): parse_polygon_string(match) } for i, match in enumerate(prediction_matches)]
    translations=[]
    # 构建 Translation 部分
    for i, (match1,match2) in enumerate(zip(polygon_matches,prediction_matches)):
      int_polygons_x,int_polygons_y=parse_polygon_int(match1)[0]
      int_predictions_polygons_x,int_predictions_polygons_y=parse_polygon_int(match2)[0]
      translations.append({"polygon{}".format(i + 1): [(int_predictions_polygons_x-int_polygons_x),(int_predictions_polygons_y-int_polygons_y)]})

         
    
    # 构建最终的 JSON
    result_json = {
        "Polygons": polygons,
        "Translation": translations,
        "Predictions": predictions
    }

    return result_json


def replace_mask(prompt,tokens):
  for token in tokens:
    prompt = prompt.replace("[MASK]", token, 1)
  return prompt



tokenizer = BertTokenizer.from_pretrained(
  'bert-large-uncased',
  padding_side="right",
  trust_remote_code=True
)
fill_mask_pipeline = pipeline(task="fill-mask", model="Sacralet/dbw-bert-large-1.1",tokenizer=tokenizer)

dataset = load_dataset('Sacralet/bertMasked_nestingDataset', split="test").select(range(200))


with open("format_inference_test.jsonl", 'w') as file:
  
  for sample in tqdm(dataset['prompt']):
    
    predictions = fill_mask_pipeline(sample)
    predicted_tokens=[]
    for prediction in predictions:
      predicted_tokens.append(prediction[0]['token_str'])

    json.dump(convert_to_json(replace_mask(sample,predicted_tokens)), file)
    file.write('\n')


