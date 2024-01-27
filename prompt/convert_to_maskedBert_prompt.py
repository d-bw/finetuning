import json
import random
from datasets import Dataset
import huggingface_hub

huggingface_hub_token = "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"
huggingface_hub.login(token=huggingface_hub_token)


def parse_polygons(polygon_list):
    return "({})".format(','.join('({},{})'.format(*map(str, item.split(','))) for item in polygon_list))


def preprocess_dataset(file):
    data=[]

    with open(file, "r", encoding="utf-8") as file:
        for line in file:
            # 从每行中加载JSON数据
            json_data = json.loads(line.strip())
            
            # 将加载的JSON数据添加到列表中
            data.append(json_data)

    return data


def convert_to_maskedBertPrompt(sample):


    message="I will complete a nesting task on a rectangular surface. The origin of the surface starts from (0,0), with the x range ending at (128,0), and the y range ending at (0,400). In this process, I will ensure each rectangle is arranged closely, maximizing the filling of the given area without overlapping by translating. After translating, the coordinates of the polygon have changed:\n"
    polygons=sample['Polygons']
    pre_polygons=sample['Predictions']
    for index,(polygon,pre_polygon) in  enumerate(zip(polygons,pre_polygons)):
        pid=f"polygon{index+1}"
        if index!=len(polygons)-1:
            message+=pid+":"+parse_polygons(polygon[pid])+"->"+parse_polygons(pre_polygon[pid])+", "
        else:
            message+=pid+":"+parse_polygons(polygon[pid])+"->"+parse_polygons(pre_polygon[pid])+"."
        

    polygons=sample['Predictions']
    max_len=len(polygons)
    #随机掩掉一个polygon的预测值
    random_mask_index = random.randint(0, max_len-1)

    polygons[random_mask_index]['polygon'+str(random_mask_index+1)]=["[MASK],[MASK]","[MASK],[MASK]","[MASK],[MASK]","[MASK],[MASK]"]
    new_sample={
        'id':sample['id'],
        'Polygons':sample['Polygons'],
        'Translation':sample['Translation'],
        'Predictions':polygons
    }


    prompt="I will complete a nesting task on a rectangular surface. The origin of the surface starts from (0,0), with the x range ending at (128,0), and the y range ending at (0,400). In this process, I will ensure each rectangle is arranged closely, maximizing the filling of the given area without overlapping by translating. After translating, the coordinates of the polygon have changed:\n"
    polygons=new_sample['Polygons']
    pre_polygons=new_sample['Predictions']

    for index,(polygon,pre_polygon) in  enumerate(zip(polygons,pre_polygons)):
        pid=f"polygon{index+1}"
        if index!=len(polygons)-1:
            prompt+=pid+":"+parse_polygons(polygon[pid])+"->"+parse_polygons(pre_polygon[pid])+", "
        else:
            prompt+=pid+":"+parse_polygons(polygon[pid])+"->"+parse_polygons(pre_polygon[pid])+"."

    return {'messages':message,'prompt':prompt}





data=preprocess_dataset("bert_nesting.jsonl")

train_dataset = Dataset.from_list(data[0:9000]+data[10000:19000]+data[20000:29000]+data[30000:39000]) 
train_dataset = train_dataset.map(convert_to_maskedBertPrompt)
train_dataset=train_dataset.select_columns(('messages','prompt')).shuffle(seed=123)

val_dataset = Dataset.from_list(data[9000:9500]+data[19000:19500]+data[29000:29500]+data[39000:39500]) 
val_dataset = val_dataset.map(convert_to_maskedBertPrompt)
val_dataset=val_dataset.select_columns(('messages','prompt'))


test_dataset = Dataset.from_list(data[9500:10000]+data[19500:20000]+data[29500:30000]+data[39500:40000]) 
test_dataset = test_dataset.map(convert_to_maskedBertPrompt)
test_dataset=test_dataset.select_columns(('messages','prompt'))


train_dataset.push_to_hub("Sacralet/bertMasked_nestingDataset", private=False, token=huggingface_hub_token,split="train")
val_dataset.push_to_hub("Sacralet/bertMasked_nestingDataset", private=False, token=huggingface_hub_token,split="validation")
test_dataset.push_to_hub("Sacralet/bertMasked_nestingDataset", private=False, token=huggingface_hub_token,split="test")