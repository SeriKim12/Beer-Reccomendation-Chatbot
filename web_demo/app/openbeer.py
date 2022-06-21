import pandas as pd
import re

beer = pd.read_csv("./app/test2.csv")
# beer = pd.read_csv("/content/drive/MyDrive/web_demo/app/test2.csv") # colab

# slot_dict 주석 및 slot_dict -> app.slot_dict 로 변경
slot_dict = {'types': ['에일'], 'abv': ['7도'], 'flavor': ['홉', '꽃'], 'taste': ['새콤달콤한']}

rcm_types_li = []
rcm_abv_li = []
rcm_flavor_li = []
rcm_taste_li = []

li_flavors = beer.loc[:, "flavor"].tolist()
li_tastes = beer.loc[:, "taste"].tolist()
li_abvs = beer.loc[:, "abv"].tolist()

# 향, 맛을 dic 형태로 전환 ex) {"맥주 이름" : ["홉", "꽃"]}
def make_set(li_slots):
    li = []
    for i in li_slots:
        i = i.split(",")
        li.append(i)
        
    li_slots = li
    li_slots = {k : v for k, v in zip(beer['kor_name'], li_slots)}
    return li_slots

# types
tmp_types_li = beer.loc[beer['types'] == slot_dict['types'][0], 'kor_name']
tmp_types_li = tmp_types_li.tolist()
rcm_types_li = tmp_types_li

# abv
li_abvs = {k : float(v) for k, v in zip(beer['kor_name'], li_abvs)}
print(li_abvs)
for abv in slot_dict['abv']:
    abv = re.sub(" ", "", abv)
    for k in li_abvs: # 도수
        if abv.endswith("도이상") and float(li_abvs[k]) >= float(abv[0]):
            rcm_abv_li.append(k)
            
        elif abv.endswith("도이하") and float(li_abvs[k]) <= float(abv[0]):
            rcm_abv_li.append(k)
            
        elif abv.endswith("도") and int(li_abvs[k]) == int(abv[0]):
            rcm_abv_li.append(k)

# flavor
for k ,v in make_set(li_flavors).items():
    for i in range(len(v)):
        for j in range(len(slot_dict['flavor'])):
            if v[i] == slot_dict['flavor'][j]:
                rcm_flavor_li.append(k)

# taste
for k ,v in make_set(li_tastes).items():
    for i in range(len(v)):
        for j in range(len(slot_dict['taste'])):
            if v[i] == slot_dict['taste'][j]:
                rcm_taste_li.append(k)


rcm_types = list(set(rcm_types_li))
rcm_abv = list(set(rcm_abv_li))
rcm_flavor = list(set(rcm_flavor_li))
rcm_taste = list(set(rcm_taste_li))

intersection = max((rcm_types + rcm_abv + rcm_flavor + rcm_taste), key= (rcm_types + rcm_abv + rcm_flavor + rcm_taste).count)
print("최종 추천 맥주 :", intersection)
