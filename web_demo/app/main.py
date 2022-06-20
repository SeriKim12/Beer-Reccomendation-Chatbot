# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import os



#from flask_ngrok import run_with_ngrok
import pickle, re
import tensorflow as tf

import sys

sys.path.append('/content/drive/MyDrive/codes/codes/')

from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer
from to_array.bert_to_array import BERTToArray
import random


############################### TODO ##########################################
# 필요한 모듈 불러오기
###############################################################################
graph = tf.compat.v1.get_default_graph()
bert_model_hub_path = '/content/drive/MyDrive/bert-module' # TODO 경로 고치기
is_bert = True

############################### TODO ##########################################
# 슬롯태깅 모델과 벡터라이저 불러오기
###############################################################################
bert_vocab_path = os.path.join(bert_model_hub_path, 'assets/vocab.korean.rawtext.list')
bert_vectorizer = BERTToArray(is_bert, bert_vocab_path)

with open(os.path.join('/content/drive/MyDrive/finetuned', 'tags_to_array.pkl'), 'rb') as handle:
  tags_vectorizer = pickle.load(handle)
  slots_num = len(tags_vectorizer.label_encoder.classes_)

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
sess = tf.compat.v1.Session(config=config)
model = BertSlotModel.load('/content/drive/MyDrive/finetuned', sess)

tokenizer = FullTokenizer(vocab_file=bert_vocab_path)

types = ['에일', 'IPA', '라거', '바이젠', '흑맥주']

abv = ['3도', '4도', '5도', '6도', '7도', '8도',
            '3도이상', '4도이상', '5도이상', '6도이상', '7도이상',
           '3도 이상', '4도 이상', '5도 이상', '6도 이상', '7도 이상',
           '4도이하', '5도이하', '6도이하', '7도이하', '8도이하',
            '4도 이하', '5도 이하', '6도 이하', '7도 이하', '8도 이하']

flavor = ['과일', '홉', '꽃', '상큼한', '커피', '스모키한']  

taste = ['단', '달달한', '달콤한', '안단', '안 단', 
              '달지 않은', '달지않은', '쓴', '씁쓸한',
              '쌉쌀한', '달콤씁쓸한', '안쓴', '안 쓴', '쓰지 않은',
              '신', '상큼한', '새콤달콤한', '시지 않은', '시지않은',
              '쓰지않은','안신', '안 신', '과일', '고소한', '구수한']

options = {'types':'종류', 'abv':'도수', 'flavor':'향', 'taste':'맛'}
dic = {i:globals()[i] for i in options}
#globals()[원하는 변수 이름] = 변수에 할당할 값 : 변수 여러개 동시 생성
# dic = {'types': types, 'abv': abv, 'flavor': flavor, 'taste': taste}

cmds = {'명령어' : [], '종류' : types, '도수' : abv, '향' : flavor, '맛' : taste}

cmds['명령어'] = [k for k in cmds]

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():

############################### TODO ##########################################
# 슬롯 사전 만들기
    #app.slot_dict = {'a_slot': None, 'b_slot':None}
  app.slot_dict = {'types' : [], 'abv' : [], 'flavor' : [], 'taste' : []}
  app.score_limit = 0.7
###############################################################################

  return render_template("index.html")
    
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

  text_arr = tokenizer.tokenize(userText)
  text_arr = [' '.join(text_arr)]
  input_ids, input_mask, segment_ids = bert_vectorizer.transform(text_arr)

  # 예측
  with graph.as_default():
    with sess.as_default():
      inferred_tags, slot_score = model.predict_slots([input_ids, input_mask, segment_ids], tags_vectorizer)

  print(text_arr)
  print(inferred_tags)
  print(slot_score)


  # 슬롯에 해당하는 텍스트를 담을 변수 설정
  slot_text = {k: "" for k in app.slot_dict}


  # # 슬롯태깅 실시
  # for i in range(0, len(inferred_tags[0])):
  #   if slot_score[0][i] >= app.score_limit:
  #     catch_slot(i, inferred_tags, text_arr, slot_text)
  #   else:
  #     print("something went wrong!")

  # 슬롯태깅 실시
  for i in range(0, len(inferred_tags[0])):     
    if slot_score[0][i] >= app.score_limit:
      if not inferred_tags[0][i] == "O":
        word_piece = re.sub("_", " ", text_arr[i])       
        slot_text[inferred_tags[0][i]] += word_piece
      #catch_slot(i, inferred_tags, text_arr, slot_text)
      #태그가 '0'가 아니면 text_arr에서 _를 지우고  slot_text에서 해당하는 태그에 단어를 담는다. 
      ##slot_text = {'abv': '', 'flavor': '', 'taste': '', 'types': ''}
    else:
      print("something went wrong!")
  print("slot_text:", slot_text)


  # 옵션의 이름과 일치하는지 검증
  for k in app.slot_dict:
    for x in dic[k]:
      x = x.lower().replace(" ", "\s*")
      m = re.search(x, slot_text[k])
      if m:
          app.slot_dict[k].append(m.group())

  empty_slot = [options[k] for k in app.slot_dict if not app.slot_dict[k]]
  filled_slot = [options[k] for k in app.slot_dict if app.slot_dict[k]]

  greetings = ['안녕', '하이', 'hi', 'hello', '헤이', '뭐해']
  idk = ['몰라', '설명해줘', '뭐야', '모르는데', '모르겠어', '알려줘', '뭔데', '몰라요', '알려주세요', '모르겠어요', '모릅니다',
        '잘 모르겠어', '잘 모르겠는데', '나 맥주 잘 몰라', '맥주 잘 모르는데', '모르겠네', '글쎄']
  mType = "맥주 종류는 '에일', 'IPA', '라거', '바이젠', '흑맥주'가 있어\n"
  ale = "에일은 풍부한 향과 진한 색이 특징이야.\n" 
  ipa = "IPA는 인디아 페일에일의 준말로, 맛이 강하고 쌉쌀한 편이지.\n" 
  lager = "라거는 탄산이 많고 가볍고 청량해.\n" 
  dark = "흑맥주는 색이 까맣고 향미가 진해.\n"
  mType2 = ale + ipa + lager + dark
  mAbv = "도수는 3도부터 8도까지 다양해.\n"
  mFlavor = "향은 '과일'향, '홉'향, '꽃'향, '상큼한' 향, '커피'향, '스모키한' 향 등이 있어.\n"
  mTaste = "맛은 '단' 맛, '달지 않은' 맛, '씁쓸한' 맛, '쓰지 않은' 맛,'신' 맛, '상큼한' 맛, '시지 않은' 맛,'과일' 맛, '구수한' 맛 등이 있지.\n"        
  answer = mType + mType2 + mAbv + mFlavor + mTaste
  
  yes = ['그래', '좋아', '좋지', '당연하지', '물론', '응', '부탁해', '네', '어', 'ㅇ']

  endings = ['quit', '종료', '그만', '멈춰', 'stop', '안마실래', '싫어', '안해', 'go away']

  message = '원하는 맥주에 대해 알려줘~~ 종류는? 도수는? 향은? 맛은? 어떤 게 좋아??~?~~~'

  if userText in greetings:
    message = '헤이~~~ 맥주 한 잔 하실??'

  elif userText in yes:
    message = '원하는 맥주에 대해 알려줘~~ 종류는? 도수는? 향은? 맛은? 어떤 게 좋아??~?~~~'

  elif userText in idk:
    message = '혹시 맥주 옵션에 대한 설명이 필요하다면 "설명"을 입력해줘'

  elif userText == '설명' : 
    message = answer

  elif userText in endings:
    message = 'Okay bye...'


  # 슬롯 1개씩만 있을 때
  elif 'types' in filled_slot :# and 'abv' not in filledSlot and 'flavor' not in filledSlot and 'taste' not in filledSlot:
    message = '접수 완료! 이제 원하는 도수, 향, 맛에 대해서도 알려줘'

  elif 'abv' in filled_slot :#  and 'type' not in filledSlot and 'flavor' not in filledSlot and 'taste' not in filledSlot:
    message = '접수 완료! 이제 원하는 종류, 향, 맛에 대해서도 알려줘'

  elif 'flavor' in filled_slot :#  and 'abv' not in filledSlot and 'type' not in filledSlot and 'taste' not in filledSlot:
    message = '접수 완료! 이제 원하는 종류, 도수, 맛에 대해서도 알려줘'

  elif 'taste' in filled_slot :#  and 'abv' not in filledSlot and 'flavor' not in filledSlot and 'type' not in filledSlot:
    message = '접수 완료! 이제 원하는 종류, 도수, 향에 대해서도 알려줘'

  # 슬롯 2개씩 : 종류도수, 종류향, 종류맛, 도수향, 도수맛, 향맛
  elif 'types' in filled_slot and 'abv' in filled_slot :#  and 'flavor' not in filledSlot and 'taste' not in filledSlot:
    message = '접수 완료! 이제 원하는 향, 맛에 대해서도 알려줘'

  elif 'types' in filled_slot and 'flavor' in filled_slot :#  and 'abv' not in filledSlot and 'taste' not in filledSlot:
    message = '접수 완료! 이제 원하는 도수, 맛에 대해서도 알려줘'

  elif 'types' in filled_slot and 'taste' in filled_slot :#  and 'abv' not in filledSlot and 'flavor' not in filledSlot:
    message = '접수 완료! 이제 원하는 도수, 향에 대해서도 알려줘'

  elif 'abv' in filled_slot and 'flavor' in filled_slot :#  and 'type' not in filledSlot and 'taste' not in filledSlot:
    message = '접수 완료! 이제 원하는 종류, 맛에 대해서도 알려줘'

  elif 'abv' in filled_slot and 'taste' in filled_slot :#  and 'type' not in filledSlot and 'flavor' not in filledSlot:
    message = '접수 완료! 이제 원하는 종류, 향에 대해서도 알려줘'

  elif 'flavor' in filled_slot and 'taste' in filled_slot :#  and 'abv' not in filledSlot and 'type' not in filledSlot:
    message = '접수 완료! 이제 원하는 종류, 도수에 대해서도 알려줘'

  # 슬롯 3개씩 : 종류도수향, 종류도수맛, 종류향맛, 도수향맛
  elif 'types' in filled_slot and 'abv' in filled_slot and 'flavor' in filled_slot :#  and 'taste' not in filledSlot:
    message = '접수 완료! 이제 원하는 맛에 대해서도 알려줘'

  elif 'types' in filled_slot and 'abv' in filled_slot and 'taste' in filled_slot :#  and 'flavor' not in filledSlot:
    message = '접수 완료! 이제 원하는 향에 대해서도 알려줘'  

  elif 'types' in filled_slot and 'flavor' in filled_slot and 'taste' in filled_slot :#  and 'abv' not in filledSlot:
    message = '접수 완료! 이제 원하는 도수에 대해서도 알려줘'  

  elif 'abv' in filled_slot and 'flavor' in filled_slot and 'taste' in filled_slot :#  and 'type' not in filledSlot:
    message = '접수 완료! 이제 원하는 종류에 대해서도 알려줘'  
  
  # 슬롯 4개
  elif 'types' in filled_slot and 'abv' in filled_slot and 'flavor' in filled_slot and 'type' in filled_slot:
    message = '네게 딱 맞을 맥주를 찾고 있는 중...'


  
  return message


############################### TODO ##########################################
# 1. 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
# 2. 추출된 슬롯 정보를 가지고 더 필요한 정보 물어보는 규칙 만들기 (if문)
    # app.slot_dict['a_slot'] = ''
    # print(app.slot_dict)

    # return 'hi' # 챗봇이 이용자에게 하는 말을 return
###############################################################################

