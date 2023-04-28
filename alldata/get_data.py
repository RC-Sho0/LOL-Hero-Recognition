import os
import json

data_path = "Eklipse/test_data/data"
with open("Eklipse/test_data/champion.json", "r") as f:
    names = json.load(f)
for cp in names['data']:
    print(cp)
    if cp != "":
        cmd = f"wget http://ddragon.leagueoflegends.com/cdn/13.8.1/img/champion/{cp}.png -P {data_path}/{cp.lower()}/"
        os.system(cmd)

print("All Done!")