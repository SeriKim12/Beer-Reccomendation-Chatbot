# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import tensorflow as tf
import os, pickle, re, sys, random
import pandas as pd

sys.path.append('/content/drive/MyDrive/codes/codes/')
from models.bert_slot_model import BertSlotModel
from to_array.bert_to_array import BERTToArray
from to_array.tokenizationK import FullTokenizer

# -----------------------------------------------------------------
# 맥주 이름&설명을 크롤링한 csv파일 읽어오기
beer = pd.read_csv('/content/drive/MyDrive/codes/web_demo/app/beer_menu.csv')

# 슬롯태깅 모델과 벡터라이저 불러오기
# ETRI에서 사전훈련한 KorBERT 체크포인트파일을 모듈로 export한 BERT 모듈 경로
bert_model_hub_path = '/content/drive/MyDrive/bert-module' 
is_bert = True


# 토큰화된 단어들을 임베딩한 단어 사전 파일
vocab_file = os.path.join(bert_model_hub_path, 'assets/vocab.korean.rawtext.list')

# 벡터라이저
bert_vectorizer = BERTToArray(is_bert, vocab_file)

# 보캡 파일로 토크나이징
tokenizer = FullTokenizer(vocab_file=vocab_file)

# loading models
load_folder_path = '/content/drive/MyDrive/finetuned'
print('Loading models ...')
if not os.path.exists(load_folder_path):
    print('Folder `%s` not exist' % load_folder_path)

# 파인튜닝을 통해 만들어진 tags_to_array.pkl 파일을 태그 벡터라이저로 사용.
with open(os.path.join(load_folder_path, 'tags_to_array.pkl'), 'rb') as handle:
  tags_vectorizer = pickle.load(handle)
  slots_num = len(tags_vectorizer.label_encoder.classes_)
 
# this line is to disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
sess = tf.compat.v1.Session(config=config)
graph = tf.compat.v1.get_default_graph()

# finetuned_epoch128로 훈련시킨 슬롯태깅모델
model = BertSlotModel.load(load_folder_path, sess)

# 슬롯 사전 단어들
types = ['에일', 'IPA', '라거', '바이젠', '흑맥주']
abv = ['3도', '4도', '5도', '6도', '7도', '8도', '3도이상', '4도이상', '5도이상', '6도이상', '7도이상',
       '3도 이상', '4도 이상', '5도 이상', '6도 이상', '7도 이상', '4도이하', '5도이하', '6도이하', '7도이하', 
       '8도이하', '4도 이하', '5도 이하', '6도 이하', '7도 이하', '8도 이하']
flavor = ['과일', '홉', '꽃', '상큼한', '커피', '스모키한']
taste = ['단', '달달한', '달콤한', '안단', '안 단', '달지 않은', '달지않은', '쓴', '씁쓸한',
          '쌉쌀한', '달콤씁쓸한', '안쓴', '안 쓴', '쓰지 않은',  '신', '상큼한', '새콤달콤한', 
          '시지 않은', '시지않은', '쓰지않은','안신', '안 신', '과일', '고소한', '구수한']

options = {'types':'종류', 'abv':'도수', 'flavor':'향', 'taste':'맛'}

dic = {i:globals()[i] for i in options}
# globals()[원하는 변수 이름] = 변수에 할당할 값 : 변수 여러개 동시 생성
# dic = {'types': types, 'abv': abv, 'flavor': flavor, 'taste': taste}

cmds = {'명령어':[], '종류':types, '도수':abv, '향':flavor, '맛':taste}
cmds["명령어"] = [k for k in cmds]
# cmds["명령어"] = ['명령어', '종류', '도수', '향', '맛']

tag_list = ['종류', '도수', '향', '맛']

greetings = ['안녕', '안녕하세요', '안뇽' '하이', 'hi', 'hello', '헤이', '뭐해']
idk = ['몰라', '설명해줘', '뭐야', '모르는데', '모르겠어', '알려줘', '뭔데', '몰라요', '알려주세요', '모르겠어요', '모릅니다',
      '잘 모르겠어', '잘 모르겠는데', '나 맥주 잘 몰라', '맥주 잘 모르는데', '모르겠네', '글쎄']
mType = "맥주 종류는 '에일', 'IPA', '라거', '바이젠', '흑맥주'가 있어.<br />\n"
ale = "에일은 풍부한 향과 진한 색이 특징이야.<br />\n" 
ipa = "IPA는 인디아 페일에일의 준말로, 맛이 강하고 쌉쌀한 편이지.<br />\n" 
lager = "라거는 탄산이 많고 가볍고 청량해.<br />\n" 
dark = "흑맥주는 색이 까맣고 향미가 진하고.<br />\n"
mType2 = ale + ipa + lager + dark
mAbv = "도수는 3도부터 8도까지 있단다.<br />\n"
mFlavor = "향은 '과일'향, '홉'향, '꽃'향, '상큼한' 향, '커피'향, '스모키한' 향 등이 있어.<br />\n"
mTaste = "맛은 '단' 맛, '달지 않은' 맛, '씁쓸한' 맛, '쓰지 않은' 맛,'신' 맛, '상큼한' 맛, '시지 않은' 맛,'과일' 맛, '구수한' 맛 등이 있지.<br />\n"        
answer = mType + mType2 + mAbv + mFlavor + mTaste

abvH = ['도수 높은', '도수높은', '도수 높고', '도수높고', '도수 쎈', '도수쎈', '강한 도수', '강한도수',
        '도수가 높고', '도수가 쎈', '도수는 높고', '도수는 높은', '도수쎄고']
abvL = ['도수 낮은', '도수낮은', '도수 낮고', '도수낮고', '도수 약한', '도수약한', '약한 도수', '약한도수',
        '도수가 낮은', '도수가 약한', '도수높은', '도수쎈']
yes = ['그래', '좋아', '좋지', '당연하지', '물론', '응', '부탁해', '네', '어', 'ㅇ']
no = ['아니', '괜찮아', '아니아니', 'ㄴㄴ', '그냥 추천해줘', '없어']
endings = ['quit', '종료', '그만', '멈춰', 'stop', '안마실래', '싫어', '안해', 'go away']

noSlot = ['맥주 마시고 싶당', '맥쥬 맥쥬', '맥주 한 잔이면 스트레스가 싸악 가셔요', 
            '원하는 맥주의 종류는? 도수는? 향은? 맛은? 어떤 게 좋니??']


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home(): # 슬롯 사전 만들기
  app.slot_dict = {'types':[], 'abv':[], 'flavor':[], 'taste':[]}
  app.score_limit = 0.8
  return render_template("index.html")



# 추천 맥주 이미지 보여주기
def showImg(name):
    return render_template('showImg.html', image_file=f'image/{name}.jpg', encoding='utf-8')
 
if __name__ == "__main__":
    app.run()


@app.route("/get")
def get_bot_response():
  userText = request.args.get('msg').strip() # 사용자가 입력한 문장
  
  if userText[0] == "!":
    try:
      li = cmds[userText[1:]]
      message = "<br />\n".join(li)
    except:
      message = "입력한 명령어가 존재하지 않습니다."

    return message

  # userText를 토크나이저를 통해 어레이로 변환
  text_arr = tokenizer.tokenize(userText)

  # 어레이로 변환된 토큰을 버트 벡터라이저로 input id, input mask, segment id로 나누기.
  input_ids, input_mask, segment_ids = bert_vectorizer.transform([" ".join(text_arr)])

  # 예측 : inferred_tags(모델이 추론한 태그 명), slots_score(슬롯일 거라고 예측한 점수) 값 구하기.
  with graph.as_default():
    with sess.as_default():
      inferred_tags, slots_score = model.predict_slots([input_ids, input_mask, segment_ids], tags_vectorizer)

  # 결과 체크
  print("text_arr:", text_arr) 
  print('input_ids:', input_ids)
  print('input_mask:', input_mask)
  print('segment_ids:', segment_ids)
  print("inferred_tags:", inferred_tags[0])
  print("slots_score:", slots_score[0])  
  

  # 슬롯에 해당하는 텍스트를 담을 변수 설정
  slot_text = {'abv': '', 'flavor': '', 'taste': '', 'types': ''}

  # 슬롯태깅 실시 
  for i in range(0, len(inferred_tags[0])):    
    if slots_score[0][i] >= app.score_limit:
      catch_slot(i, inferred_tags, text_arr, slot_text)
      # catch_slot : inferred_tags가 '0'가 아니면 text_arr에서 _를 지우고 slot_text에서 해당하는 태그에 단어를 담는다.

    else: # 슬롯 스코어가 스코어 리밋인 0.8보다 낮을 때
      print("something went wrong!") 
  print("slot_text:", slot_text)
  print('text_arr after slot tagging :', text_arr)

  
  # 에일을 잡지 못해서 slot_text에 직접 추가
  if '에일' in userText:
    slot_text['types'] = '에일'

  # 도수 높은~ 과 비슷한 뉘앙스의 말 = 6도이상 으로 태깅
  for a in abvH:
    if a in userText:
      slot_text['abv'] = '6도이상'

  
  # 도수 낮은~ 과 비슷한 뉘앙스의 말 = 4도이하 로 태깅
  for a in abvL:
    if a in userText:
      slot_text['abv'] = '4도이하'

  # 옵션의 이름과 일치하는지 검증
  for k in app.slot_dict:  # k : 'types','abv','flavor','taste' 
    for x in dic[k]:
      x = x.lower().replace(" ", "\s*") # 대문자를 소문자로, 공백이 있는 부분을 공백이 있거나 없거나로 변경.
      m = re.search(x, slot_text[k])

      if m:
        app.slot_dict[k].append(m.group())
  
  print('app.slot_dict : ', app.slot_dict)


  # 안~~ 인 형용사를 re.search가 단/쓴/신 등의 원본 단어도 찾아버려서
  taste_v = ['안 쓴', '안쓴', '안 신', '안신', '안단', '안 단']
  for i in taste_v :
    if i in slot_text['taste'] :
      del(app.slot_dict['taste'][0]) #app.slot_dict['taste'][0] 은 항상 쓴/단/신 등이므로

  print("app.slot_dict :", app.slot_dict) 

  abv_v = ['3도이상', '4도이상', '5도이상', '6도이상', '7도이상', '8도이하', '4도이하', '5도이하', '6도이하', '7도이하']
  for i in abv_v :
    if i in slot_text['abv'] :
      del(app.slot_dict['abv'][0])
  print("app.slot_dict :", app.slot_dict)    
  


  #options = {'beer_types':'종류', 'beer_abv':'도수', 'beer_flavor':'향', 'beer_taste':'맛'}
  empty_slot = [options[k] for k in app.slot_dict if not app.slot_dict[k]]
  filled_slot = [options[k] for k in app.slot_dict if app.slot_dict[k]]    


  print("empty_slot :", empty_slot)
  print("filled_slot :", filled_slot)

  if userText in greetings:
    message = '헤이~~~ 맥주 한 잔 하실??'
    return message

  elif userText in yes:
    message = '원하는 맥주에 대해 알려줘~~ 종류는? 도수는? 향은? 맛은? 어떤 게 좋아??~?~~~'
    return message

  elif userText in idk:
    message = '혹시 맥주 옵션에 대한 설명이 필요하다면 "설명"을, 아니라면 "x"를 입력해줘'
    return message

  elif userText == '설명' : 
    message = answer
    return message

  elif userText == 'x' : 
    message = "원하는 맥주에 대해 알려줘!"
    return message

  elif userText in endings:
    message = 'Okay bye...'
    init_app(app)
    return message

  elif userText in ['고마워', 'ㄱㅅ', '땡큐', '감사합니다', '감사해요']:
    message = '고맙긴 뭘ㅎㅎ 그럼 즐맥하라구~!!'
    return message

  # for i in ['에일', 'IPA', '라거', '바이젠', '흑맥주', 'ipa']:
  #   if 'types' in inferred_tags[0] and i not in userText:
  #     #app.slot_dict['types'] = None
  #     del slot_text['types']
  #     message = f'네가 찾는 건 없네ㅠㅠ {mType}'
  #     init_app(app)
  #     return message
  
  # for i in ['3도', '4도', '5도', '6도', '7도', '8도']:
  #   if 'abv' in inferred_tags[0] and i not in userText:
  #     #app.slot_dict['abv'] = None
  #     del slot_text['abv']
  #     message = f'네가 찾는 건 없어ㅠㅠ {mAbv}'
  #     init_app(app)
  #     return message
  
  # for i in taste:
  #   if 'taste' in inferred_tags[0] and i not in userText:
  #     #app.slot_dict['taste'] = None
  #     del slot_text['taste']
  #     message = f'네가 찾는 건 없네ㅠㅠ {mTaste}'
  #     init_app(app)
  #     return message

  # for i in flavor:
  #   if 'flavor' in inferred_tags[0] and i not in userText:
  #     #app.slot_dict['flavor'] = None
  #     del slot_text['flavor']
  #     message = f'네가 찾는 건 없네ㅠㅠ {mFlavor}'
  #     init_app(app)
  #     return message



  


  # 추천할 슬롯별 맥주 이름 목록을 담을 빈 리스트 생성
  rcm_types_li = []
  rcm_abv_li = []
  rcm_flavor_li = []
  rcm_taste_li = []

  
  # beer 데이터 프레임에서 불러온 종류/향/맛/도수 특징에 대한 리스트 생성
  li_types = beer.loc[:, 'types'].tolist()
  li_flavors = beer.loc[:, "flavor"].tolist()
  li_tastes = beer.loc[:, "taste"].tolist()
  li_abvs = beer.loc[:, "abv"].tolist()

  # 입력받은 types 슬롯 단어에 해당하는 맥주 이름 불러오기
  for k ,v in make_set(li_types).items():
    for i in range(len(v)):
      for j in range(len(app.slot_dict['types'])):
        if v[i] == app.slot_dict['types'][j]:
          rcm_types_li.append(k)

  # 입력받은 abv 슬롯 단어에 해당하는 맥주 이름 불러오기
  li_abvs = {k : float(v) for k, v in zip(beer['kor_name'], li_abvs)}

  for abv in app.slot_dict['abv']:
    abv = re.sub(" ", "", abv)
    for k in li_abvs: # 도수
      if abv.endswith("도이상") and float(li_abvs[k]) >= float(abv[0]):
        rcm_abv_li.append(k)
          
      elif abv.endswith("도이하") and float(li_abvs[k]) <= float(abv[0]):
        rcm_abv_li.append(k)
          
      elif abv.endswith("도") and int(li_abvs[k]) == int(abv[0]):
        rcm_abv_li.append(k)

  # 입력받은 flavor 슬롯 단어에 해당하는 맥주 이름 불러오기
  for k ,v in make_set(li_flavors).items():
    for i in range(len(v)):
      for j in range(len(app.slot_dict['flavor'])):
        if v[i] == app.slot_dict['flavor'][j]:
          rcm_flavor_li.append(k)

  # 입력받은 taste 슬롯 단어에 해당하는 맥주 이름 불러오기
  for k ,v in make_set(li_tastes).items():
    for i in range(len(v)):
      for j in range(len(app.slot_dict['taste'])):
        #if v[i] == app.slot_dict['taste'][j]:
        if app.slot_dict['taste'][j] in v[i]:
          rcm_taste_li.append(k)

  rcm_types = list(set(rcm_types_li))
  rcm_abv = list(set(rcm_abv_li))
  rcm_flavor = list(set(rcm_flavor_li))
  rcm_taste = list(set(rcm_taste_li))
  
  print("rcm_types :", rcm_types) 
  print("rcm_abv :", rcm_abv)
  print("rcm_flavor :", rcm_flavor)
  print("rcm_taste :", rcm_taste)

  # 최종 추천 제품
  rcm_merge = rcm_types + rcm_abv + rcm_flavor + rcm_taste # 위 리스트를 합친 것
  
  # 키값
  rcm_merge_key = rcm_merge.count
  intersection_beer = max(rcm_merge, key= rcm_merge_key) # 가장 많이 반복 되는 맥주
  intersection_idx = rcm_merge.index(intersection_beer) # 가장 많이 반복 되는 맥주의 index

  if intersection_idx == 0 and rcm_merge.count(intersection_beer) == 1: # 입력된 슬롯이 하나일 경우
    rcm_idx = intersection_idx                        # 가장 많이 반복되는 맥주의 index값에,
    rcm_idx += random.randrange(0, len(rcm_merge))  # 0부터 최종 추천 제품 리스트 길이만큼의 범위에서 랜덤으로 숫자를 더한 후
    fin_rcm_beer = rcm_merge[rcm_idx]               # 최종 추천 제품 리스트에서 그 숫자만큼의 인덱스 번호를 가진 맥주를 추천함
                          # 이렇게 강제적으로 인덱스를 바꾸지 않으면 항상 rcm_merge의 0번 인덱스가 intersection_beer라고 인식되기 때문

  else:
    fin_rcm_beer = intersection_beer

  print("최종 추천 제품 :", fin_rcm_beer)


  
  # 종류, 도수, 향, 맛 슬롯이 하나도 채워지지 않은 문장일 경우 = noSlot에서 랜덤으로 출력.
  if ('종류' in empty_slot and '도수' in empty_slot and '향' in empty_slot and '맛' in empty_slot):
    message = random.choice(noSlot)
    return message

  # 슬롯이 하나 이상 채워질 경우, inferred_tag가 'O'이 아니면 msg_li에 추가
  elif ('종류' in filled_slot or '도수' in filled_slot or '향' in filled_slot or '맛' in filled_slot):
    msg_li = []
    for i in range(0, len(inferred_tags[0])):
      if not inferred_tags[0][i] == "O":
        msg_li.append(slot_text[inferred_tags[0][i]])
        break
    message = chatbot_msg(msg_li, slot_text)
    # message = "{} 말이지? <br />\n더 고려할 사항이 있니?".format(msg_li)

    if userText in ['응', '네', '있어']: 
      ask_msg = "어떤 걸 찾고 있어?"
      return ask_msg

    elif userText in no:
      last_msg = f"알았어! 널 위한 맥주는 바로!! {fin_rcm_beer}!!" + showImg(fin_rcm_beer)
      init_app(app)
      return last_msg

    return message

          
# 종류, 향, 맛 슬롯 단어에 해당하는 맥주를 dic 형태로 전환 ex) {"맥주 이름" : ["홉", "꽃"]}
def make_set(li_slots):
  li = []
  for i in li_slots:
    i = i.split(",")
    li.append(i)
      
  li_slots = li
  li_slots = {k : v for k, v in zip(beer['kor_name'], li_slots)}
  return li_slots

# 예측된 태그가 O가 아니라면 정규표현식으로 '_' 를 제거    
def catch_slot(i, inferred_tags, text_arr, slot_text):
  if not inferred_tags[0][i] == "O":
    word_piece = re.sub("_", "", text_arr[i])
    slot_text[inferred_tags[0][i]] += word_piece
    
# infered_tags = ['O', 'abv', 'abv', 'O', 'type', 'type', 'type', 'O', 'O', 'O', 'O', 'O', 'flavor', 'flavor', 'flavor', 'flavor', 'O', 'O']
#text_arr = ['나는_', '7', '도_', '넘는_', '흑', '맥', '주로_', '주', '문', '하고_', '싶', '어_', '스', '모', '키', '한_', '걸', '로_']
#slot_text = {'beer_abv': '7도', 'beer_flavor': '', 'beer_taste': '', 'beer_types': '흑맥주'}

def init_app(app):
  app.slot_dict = {'types': [], 'abv':[], 'flavor':[], 'taste':[]}

# 공백, 중복 제거 함수. 과일맛, 과일향 둘 다 출력하기 위해 각각 조건문으로 '향', '맛'을 포함시킴.
def chatbot_msg(msg_li, slot_text):
  for k in app.slot_dict:
    if k == 'flavor':
      for i in range(len(app.slot_dict[k])):
        if slot_text[k] == app.slot_dict[k][i]:
          app.slot_dict[k][i] += ' 향'
    elif k == 'taste':
      for i in range(len(app.slot_dict[k])):
        if slot_text[k] == app.slot_dict[k][i]:
          app.slot_dict[k][i] += ' 맛'
    msg_li.extend(app.slot_dict[k])

  for i in range(len(msg_li)):
    msg_li[i] = msg_li[i].strip() # msg_li에서 공백 제거
  del msg_li[0]
  msg_li = list(set(msg_li)) # 중복 제거
  message = "{} 말이지? <br />\n더 고려할 사항이 있니?".format(msg_li)
  return message

  
