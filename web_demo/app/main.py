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
#model._make_predict_function()
tokenizer = FullTokenizer(vocab_file=bert_vocab_path)


def catch_slot(i, inferred_tags, text_arr, slot_text):
  if not inferred_tags[0][i] == "O":
    word_piece = re.sub("_", " ", text_arr[i])
    if word_piece == 'ᆫ':
      word = slot_text[inferred_tags[0][i]]
      slot_text[inferred_tags[0][i]] = word[:-1]+chr(ord(word[-1])+4)
    else:    
      slot_text[inferred_tags[0][i]] += word_piece



# def exception_handling(slots, empty, filled):
#   for slot in slots:
#     if slot in empty:
#       message = f'원하는 {slot}에 대해 알려줘!'
#     elif slot in filled:
#       message = f'알았어 {slot} 정보 수집 완료'

#   return message




beer_types = ['에일', 'IPA', '라거', '바이젠', '흑맥주']

beer_abv = ['3도', '4도', '5도', '6도', '7도', '8도',
            '3도이상', '4도이상', '5도이상', '6도이상', '7도이상',
           '3도 이상', '4도 이상', '5도 이상', '6도 이상', '7도 이상',
           '4도이하', '5도이하', '6도이하', '7도이하', '8도이하',
            '4도 이하', '5도 이하', '6도 이하', '7도 이하', '8도 이하']

beer_flavor = ['과일', '홉', '꽃', '상큼한', '커피', '스모키한']  

beer_taste = ['단', '달달한', '달콤한', '안단', '안 단', 
              '달지 않은', '달지않은', '쓴', '씁쓸한',
              '쌉쌀한', '달콤씁쓸한', '안쓴', '안 쓴', '쓰지 않은',
              '신', '상큼한', '새콤달콤한', '시지 않은', '시지않은',
              '쓰지않은','안신', '안 신', '과일', '고소한', '구수한']

options = {'type' : '종류', 'abv' : '도수', 'flavor' : '향', 'taste' : '맛'}


cmds = {'명령어' : [], '종류' : beer_types, '도수' : beer_abv, '향' : beer_flavor, '맛' : beer_taste}

cmds['명령어'] = [cmd for cmd in cmds]

app = Flask(__name__)
#run_with_ngrok(app) # 코랩 실행 시
app.static_folder = 'static'

@app.route("/")
def home():

############################### TODO ##########################################
# 슬롯 사전 만들기
    #app.slot_dict = {'a_slot': None, 'b_slot':None}
  app.slot_dict = {'type' : [], 'abv' : [], 'flavor' : [], 'taste' : []}
  app.score_limit = 0.7
###############################################################################

  return render_template("index.html")
    
@app.route("/get")
def get_bot_response():
  

  userText = request.args.get('msg').strip() # 사용자가 입력한 문장


  text_arr = tokenizer.tokenize(userText)
  input_ids, input_mask, segment_ids = bert_vectorizer.transform(' '.join(text_arr))
  #inferred_tags, slot_score = model.predict_slots([input_ids, input_mask, segment_ids], tags_vectorizer)

  # 예측
  with graph.as_default():
    with sess.as_default():
      inferred_tags, slot_score = model.predict_slots(
        [input_ids, input_mask, segment_ids], tags_vectorizer
            )

  # 슬롯에 해당하는 텍스트를 담을 변수 설정
  slot_text = {k: "" for k in app.slot_dict}

  
  # 슬롯태깅 실시
  for i in range(0, len(inferred_tags[0])):
    if slot_score[0][i] >= app.score_limit:
      catch_slot(i, inferred_tags, text_arr, slot_text)
    else:
      print("something went wrong!")

  empty_slot = [options[j] for j in app.slot_dict if not app.slot_dict[j]]
  filled_slot = [options[m] for m in app.slot_dict if app.slot_dict[m]]

  


  # if userText in ['안녕', '하이', 'hi', 'hello']:
  #   message = '헤이~~~'

  # userText 안에 greetings 문자가 들어있으면 message = 헤이~~~ 맥주 한 잔 하실??
  greetings = ['안녕', '하이', 'hi', 'hello', '헤이', '뭐해']
  #if [txt for txt in greetings if txt in userText] is not None:
  if userText in greetings:
    message = '헤이~~~ 맥주 한 잔 하실??'

  # exception_handling(['종류', '도수', '향', '맛'], empty_slot, filled_slot)

  if ['종류', '도수', '향', '맛'] in empty_slot:
    mA = '원하는 맥주에 대해 알려줘~~ 종류는? 도수는? 향은? 맛은? 어떤 게 좋아??~?~~~'
    mB = '혹시 맥주 옵션에 대한 설명이 필요하니?'
    mType = "맥주 종류는 '에일', 'IPA', '라거', '바이젠', '흑맥주'가 있어"
    mAbv = "도수는 3도부터 8도까지 다양해"
    mFlavor = "향은 '과일'향, '홉'향, '꽃'향, '상큼한' 향, '커피'향, '스모키한' 향 등이 있어"
    mTaste = "맛은 '단' 맛, '달지 않은' 맛, '씁쓸한' 맛, '쓰지 않은' 맛,'신' 맛, '상큼한' 맛, '시지 않은' 맛,'과일' 맛, '구수한' 맛 등이 있지"


  # userText 안에 idk 문자가 들어있으면 message = mB
  idk = ['몰라', '설명해줘', '뭐야', '모르는데', '모르겠어', '알려줘', '뭔데', '몰라요', '알려주세요', '모르겠어요', '모릅니다']
  #if [txt for txt in idk if txt in userText] is not None:
  if userText in idk:
    message = '에일은 뭐고 바이젠은 뭐고 흑맥주는 뭐고 IPA는 뭐야 블라블라'


  elif ['종류'] in filled_slot:
    if ['도수', '향', '맛'] in empty_slot:
      message = f'접수 완료! 이제 원하는 도수, 향, 맛에 대해서도 알려줘'
    elif ['도수'] in filled_slot and ['향', '맛'] in empty_slot:
      message = '오케이! 원하는 향과 맛은?'    
    elif ['향'] in filled_slot and ['도수', '맛'] in empty_slot:
      message = '오케이! 원하는 도수와 맛은?'
    elif ['맛'] in filled_slot and ['도수', '향'] in empty_slot:
      message = '오케이! 원하는 도수와 향은?'
    elif ['도수', '향'] in filled_slot and ['맛'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 맛은?'    
    elif ['도수', '맛'] in filled_slot and ['향'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 향은?'
    elif ['향', '맛'] in filled_slot and ['도수'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 도수는?'


  elif ['도수'] in filled_slot:
    if ['종류', '향', '맛'] in empty_slot:
      message = f'접수 완료! 이제 원하는 종류, 향, 맛에 대해서도 알려줘'
    elif ['종류'] in filled_slot and ['향', '맛'] in empty_slot:
      message = '오케이! 원하는 향과 맛은?'    
    elif ['향'] in filled_slot and ['종류', '맛'] in empty_slot:
      message = '오케이! 원하는 종류와 맛은?'
    elif ['맛'] in filled_slot and ['종류', '향'] in empty_slot:
      message = '오케이! 원하는 종류와 향은?'
    elif ['종류', '향'] in filled_slot and ['맛'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 맛은?'    
    elif ['종류', '맛'] in filled_slot and ['향'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 향?'
    elif ['맛', '향'] in filled_slot and ['종류'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 종류는?'


  elif ['향'] in filled_slot:
    if ['종류', '도수', '맛'] in empty_slot:
      message = f'접수 완료! 이제 원하는 종류, 도수, 맛에 대해서도 알려줘'
    elif ['종류'] in filled_slot and ['도수', '맛'] in empty_slot:
      message = '오케이! 원하는 도수와 맛은?'    
    elif ['도수'] in filled_slot and ['종류', '맛'] in empty_slot:
      message = '오케이! 원하는 종류와 맛은?'
    elif ['맛'] in filled_slot and ['종류', '도수'] in empty_slot:
      message = '오케이! 원하는 종류와 도수는?'
    elif ['종류', '도수'] in filled_slot and ['맛'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 맛은?'    
    elif ['종류', '맛'] in filled_slot and ['도수'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 도수는?'
    elif ['도수', '맛'] in filled_slot and ['종류'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 종류는?'


  elif ['맛'] in filled_slot:
    if ['종류', '도수', '향'] in empty_slot:
      message = f'접수 완료! 이제 원하는 종류, 도수, 향에 대해서도 알려줘'
    elif ['종류'] in filled_slot and ['도수', '향'] in empty_slot:
      message = '오케이! 원하는 도수와 향은?'    
    elif ['도수'] in filled_slot and ['종류', '향'] in empty_slot:
      message = '오케이! 원하는 종류와 향은?'
    elif ['향'] in filled_slot and ['종류', '도수'] in empty_slot:
      message = '오케이! 원하는 종류와 도수는?'
    elif ['종류', '도수'] in filled_slot and ['향'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 향은?'    
    elif ['종류', '향'] in filled_slot and ['도수'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 도수는?'
    elif ['도수', '향'] in filled_slot and ['종류'] in empty_slot:
      message = '좋아 이제 마지막으로 원하는 종류는?'


  elif ['종류', '도수', '향', '맛'] in filled_slot:
    message = '접수 완료!! 네게 딱 맞는 맥주를 찾고 있는 중...'

  # if userText in ['quit', '종료', '그만', '멈춰', 'stop', '안마실래', '싫어', '안해', 'go away']:
  #   message = 'Okay bye...'

  # userText 안에 endings 문자가 들어있으면 message = Okay bye
  endings = ['quit', '종료', '그만', '멈춰', 'stop', '안마실래', '싫어', '안해', 'go away']
  #if [txt for txt in endings if txt in userText] is not None: 안녕의 '안'도 엔딩으로 잡아서 폐기
  if userText in endings:
    message = 'Okay bye...'


  return message



    

  



############################### TODO ##########################################
# 1. 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
# 2. 추출된 슬롯 정보를 가지고 더 필요한 정보 물어보는 규칙 만들기 (if문)
    # app.slot_dict['a_slot'] = ''
    # print(app.slot_dict)

    # return 'hi' # 챗봇이 이용자에게 하는 말을 return
###############################################################################

