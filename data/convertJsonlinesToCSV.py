import json
import pandas as pd

data = []
with open('Arma_3.jsonlines') as data1:
    for obj in data1:
        data.append(json.loads(obj))

with open('Counter_Strike_Global_Offensive.jsonlines') as data2:
    for obj in data2:
        data.append(json.loads(obj))

with open('Counter_Strike.jsonlines') as data3:
    for obj in data3:
        data.append(json.loads(obj))

with open('Dota_2.jsonlines') as data4:
    for obj in data4:
        data.append(json.loads(obj))
    
with open('Football_Manager_2015.jsonlines') as data5:
    for obj in data5:
        data.append(json.loads(obj))

with open('Garrys_Mod.jsonlines') as data6:
    for obj in data6:
        data.append(json.loads(obj))

with open('Grand_Theft_Auto_V.jsonlines') as data7:
    for obj in data7:
        data.append(json.loads(obj))

with open('Sid_Meiers_Civilization_5.jsonlines') as data8:
    for obj in data8:
        data.append(json.loads(obj))

with open('Team_Fortress_2.jsonlines') as data9:
    for obj in data9:
        data.append(json.loads(obj))

with open('The_Elder_Scrolls_V.jsonlines') as data10:
    for obj in data10:
        data.append(json.loads(obj))

with open('Warframe.jsonlines') as data11:
    for obj in data11:
        data.append(json.loads(obj))

df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)