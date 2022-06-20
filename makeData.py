import random
import re

# 슬롯 : 종류, 도수, 맛, 향 
beer_types = ['/types;에일/', '/types;IPA/', '/types;라거/', '/types;바이젠/', '/types;흑맥주/']

beer_abv = ['/abv;3도/', '/abv;4도/', '/abv;5도/', '/abv;6도/', '/abv;7도/', '/abv;8도/',
            '/abv;3도이상/', '/abv;4도이상/', '/abv;5도이상/', '/abv;6도이상/', '/abv;7도이상/',
           '/abv;3도 이상/', '/abv;4도 이상/', '/abv;5도 이상/', '/abv;6도 이상/', '/abv;7도 이상/',
           '/abv;4도이하/', '/abv;5도이하/', '/abv;6도이하/', '/abv;7도이하/', '/abv;8도이하/',
            '/abv;4도 이하/', '/abv;5도 이하/', '/abv;6도 이하/', '/abv;7도 이하/', '/abv;8도 이하/']
beer_flavor = ['/flavor;과일/', '/flavor;홉/', '/flavor;꽃/', '/flavor;상큼한/', '/flavor;커피/', '/flavor;스모키한/']  
     

beer_taste = ['/taste;단/', '/taste;달달한/', '/taste;달콤한/', '/taste;안단/', '/taste;안 단/', 
              '/taste;달지 않은/', '/taste;달지않은/', '/taste;쓴/', '/taste;씁쓸한/',
              '/taste;쌉쌀한/', '/taste;달콤씁쓸한/', '/taste;안쓴/', '/taste;안 쓴/', '/taste;쓰지 않은/',
              '/taste;신/', '/taste;상큼한/', '/taste;새콤달콤한/', '/taste;시지 않은/', '/taste;시지않은/',
            '/taste;쓰지않은/','/taste;안신/', '/taste;안 신/', '/taste;과일/', '/taste;고소한/', '/taste;구수한/']

cnt = 85
slots = []
append_text = slots.append        

# 문장 불러오기 함수
def slots_txt(filename):
    
    f = open(filename, 'r', encoding = 'utf-8')
    lines = f.readlines()
    
    rnd_types = random.choice(beer_types)
    rnd_abv = random.choice(beer_abv)
    rnd_flavor = random.choice(beer_flavor)
    rnd_taste = random.choice(beer_taste)

    for line in lines:
        # txt 파일 내 주석 제거
        if "#" in line:
            line = line.replace(line,"")
            
        if "{1}" or "{2}" or "{3}" or "{4}" in line:
            
            line = line.replace("{1}", rnd_types).strip()
            line = line.replace("{2}", rnd_abv).strip()
            line = line.replace("{3}", rnd_flavor).strip()
            line = line.replace("{4}", rnd_taste).strip()
            
            
            #(ord(rnd_types[-2])-ord('가')) % 28 == 4 or 8: # 4면 받침이 ㄴ, 8이면 ㄹ
            if 44032 <= ord(rnd_types[-2]) <= 55203 and (ord(rnd_types[-2]) - ord("가")) % 28 > 0:   
                line = line.replace(rnd_types+'면', rnd_types+'이면')
                line = line.replace(rnd_types+'가', rnd_types+'이')
                line = line.replace(rnd_types+'였으면', rnd_types+'이었으면')

            
            if (ord(rnd_flavor[-2])-ord('가')) % 28 != 4 :
                for reg in ['거로', '거', '걸로', '게', '건데', '으로', '것', ' 거로', ' 거', ' 걸로', ' 게', ' 건데', ' 으로', ' 것']:
                    line = line.replace(rnd_flavor+reg, rnd_flavor+'향 나는'+''+reg)
             
            for regex in ['으로', ' 으로']:
                line = line.replace(rnd_flavor+regex, rnd_flavor+' 향으로')
                
            if (ord(rnd_flavor[-2])-ord('가')) % 28 == 4:
                line = line.replace(rnd_flavor+'로', rnd_flavor+'으로')
                
            
            # 정규 표현식
            line = re.sub(re.compile(r'/types;흑맥주/\s*맥주+\s*'), '/types;흑맥주/', line)
            line = re.sub(re.compile(r'/types;바이젠/\s*로'), '/types;바이젠/으로', line)
            
            line = re.sub(re.compile(r'도\s*이상/로'), '도 이상/으로', line)
            line = re.sub(re.compile(r'도\s*이상/고'), '도 이상/이고', line)
            
            line = re.sub(re.compile(r'/taste;과일/\s*(거|걸로|게|건데|으로|것)'), '/taste;과일/맛 나는 거', line)
            
            slots.append(line)
                        
    f.close()

# 슬롯이 없는 문장 불러오기
slots_txt("./slot_txt/slot_de.txt")

# 문장 틀 불러오기
for i in range(cnt):
    slots_txt("./slot_txt/slot_1.txt")
    slots_txt("./slot_txt/slot_2.txt")
    slots_txt("./slot_txt/slot_3.txt")
    slots_txt("./slot_txt/slot_4.txt")
    
# 중복 제거    
slots_unique = set(slots)
slots = list(slots_unique)

# data.txt 에 저장
file = 'data.txt'
with open(file, 'w+', encoding = 'UTF-8') as f:
    f.write('\n'.join(slots))

print("")    
print("data.txt complete")
print("sentence cnt :", len(slots))
