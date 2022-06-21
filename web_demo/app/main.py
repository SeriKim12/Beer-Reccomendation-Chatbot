# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import tensorflow as tf
import os, pickle, re, sys
import pandas as pd

sys.path.append('/content/drive/MyDrive/codes/codes/')
from models.bert_slot_model import BertSlotModel
from to_array.bert_to_array import BERTToArray
from to_array.tokenizationK import FullTokenizer

# -----------------------------------------------------------------


# 슬롯태깅 모델과 벡터라이저 불러오기

bert_model_hub_path = '/content/drive/MyDrive/bert-module' # TODO 경로 고치기
# pretrained BERT 모델을 모듈로 export - ETRI에서 사전훈련한 BERT의 체크포인트를 가지고 만든 BERT 모듈
is_bert = True


# 토큰화된 단어들에 숫자 매겨놓은 것
vocab_file = os.path.join(bert_model_hub_path, 'assets/vocab.korean.rawtext.list')

# 벡터라이저
bert_vectorizer = BERTToArray(is_bert, vocab_file)

# 보캡 파일로 토크나이징
tokenizer = FullTokenizer(vocab_file=vocab_file)


load_folder_path = '/content/drive/MyDrive/finetuned'
# loading models
print('Loading models ...')
if not os.path.exists(load_folder_path):
    print('Folder `%s` not exist' % load_folder_path)

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
          '시지 않은', '시지않은', '쓰지않은/','안신', '안 신', '과일', '고소한', '구수한']

options = {'types':'종류', 'abv':'도수', 'flavor':'향', 'taste':'맛'}
dic = {i:globals()[i] for i in options}
# globals()[원하는 변수 이름] = 변수에 할당할 값 : 변수 여러개 동시 생성
# dic = {'types': types, 'abv': abv, 'flavor': flavor, 'taste': taste}

cmds = {'명령어':[],
        '종류':types,
        '도수':abv,
        '향':flavor,
        '맛':taste}
cmds["명령어"] = [k for k in cmds]
# cmds["명령어"] = ['명령어', '종류', '도수', '향', '맛']
tag_list = ['종류', '도수', '향', '맛']

greetings = ['안녕', '안녕하세요', '안뇽' '하이', 'hi', 'hello', '헤이', '뭐해']
idk = ['몰라', '설명해줘', '뭐야', '모르는데', '모르겠어', '알려줘', '뭔데', '몰라요', '알려주세요', '모르겠어요', '모릅니다',
      '잘 모르겠어', '잘 모르겠는데', '나 맥주 잘 몰라', '맥주 잘 모르는데', '모르겠네', '글쎄']
mType = "맥주 종류는 '에일', 'IPA', '라거', '바이젠', '흑맥주'가 있어\n"
ale = "에일은 풍부한 향과 진한 색이 특징이야.\n" 
ipa = "IPA는 인디아 페일에일의 준말로, 맛이 강하고 쌉쌀한 편이지.\n" 
lager = "라거는 탄산이 많고 가볍고 청량해.\n" 
dark = "흑맥주는 색이 까맣고 향미가 진하고.\n"
mType2 = ale + ipa + lager + dark
mAbv = "도수는 3도부터 8도까지 있단다.\n"
mFlavor = "향은 '과일'향, '홉'향, '꽃'향, '상큼한' 향, '커피'향, '스모키한' 향 등이 있어.\n"
mTaste = "맛은 '단' 맛, '달지 않은' 맛, '씁쓸한' 맛, '쓰지 않은' 맛,'신' 맛, '상큼한' 맛, '시지 않은' 맛,'과일' 맛, '구수한' 맛 등이 있지.\n"        
answer = mType + mType2 + mAbv + mFlavor + mTaste

abvH = ['도수 높은', '도수높은', '도수 높고', '도수높고', '도수 쎈', '도수쎈', '강한 도수', '강한도수',
        '도수가 높고', '도수가 쎈', '도수는 높고', '도수는 높은', '도수쎄고']
abvL = ['도수 낮은', '도수낮은', '도수 낮고', '도수낮고', '도수 약한', '도수약한', '약한 도수', '약한도수',
        '도수가 낮은', '도수가 약한', '도수높은', '도수쎈']
yes = ['그래', '좋아', '좋지', '당연하지', '물론', '응', '부탁해', '네', '어', 'ㅇ']
no = ['아니', '괜찮아', '아니아니', 'ㄴㄴ', '그냥 추천해줘', '없어']
endings = ['quit', '종료', '그만', '멈춰', 'stop', '안마실래', '싫어', '안해', 'go away']


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home(): # 슬롯 사전 만들기
  app.slot_dict = {'types':[], 'abv':[], 'flavor':[], 'taste':[]}
  app.score_limit = 0.8
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
  input_ids, input_mask, segment_ids = bert_vectorizer.transform([" ".join(text_arr)])

  # 예측
  with graph.as_default():
    with sess.as_default():
      inferred_tags, slots_score = model.predict_slots(        
        [input_ids, input_mask, segment_ids], tags_vectorizer
      )

  # 결과 체크
  print("text_arr:", text_arr) 
  print("inferred_tags:", inferred_tags[0])
  print("slots_score:", slots_score[0])   

  # 슬롯에 해당하는 텍스트를 담을 변수 설정
  #slot_text = {k: "" for k in app.slot_dict}
  #app.slot_dict = {'types':[], 'abv':[], 'flavor':[], 'taste':[]}
  slot_text = {'abv': '', 'flavor': '', 'taste': '', 'types': ''}

  # # 슬롯태깅 실시 : 태그가 '0'가 아니면 text_arr에서 _를 지우고  slot_text에서 해당하는 태그에 단어를 담는다
  # for i in range(0, len(inferred_tags[0])):   
  #   if slots_score[0][i] >= app.score_limit:
  #     catch_slot(i, inferred_tags, text_arr, slot_text)
  #     ##slot_text = {'abv': '', 'flavor': '', 'taste': '', 'types': ''}
  #   else:
  #     print("something went wrong!")
  # print("slot_text:", slot_text)


  # # 메뉴판의 이름과 일치하는지 검증
  # for k in app.slot_dict:
  # # k : 'types','abv','flavor','taste' 
  #     for x in dic[k]:
  #     # {'types': [types], 'abv': [abv], 'flavor': [flavor], 'taste': [taste]}  
  #       x = x.lower().replace(" ", "\s*")
  #       m = re.search(x, slot_text[k])
  #       if m:
  #         app.slot_dict[k].append(m.group())
  # print("app.slot_dict :", app.slot_dict)        

  # 슬롯태깅 실시 : 태그가 '0'가 아니면 text_arr에서 _를 지우고  slot_text에서 해당하는 태그에 단어를 담는다. 
  for i in range(0, len(inferred_tags[0])):     
    if slots_score[0][i] >= app.score_limit:
      if not inferred_tags[0][i] == "O":
        word_piece = re.sub("_", " ", text_arr[i])       
        slot_text[inferred_tags[0][i]] += word_piece
        #slot_text = {'abv': '', 'flavor': '', 'taste': '', 'types': ''}
    else:
      print("something went wrong!")
  print("slot_text:", slot_text)


  # 옵션의 이름과 일치하는지 검증
  for k in app.slot_dict:
    for x in dic[k]:
      x = x.lower().replace(" ", "\s*")
      m = re.search(x, slot_text[k])
      #print('x:', x)
      #print('m:', m)
      if m:
          app.slot_dict[k].append(m.group())

  # # 에일 쑤셔박기!!!
  # if 'types' in inferred_tags[0] and '에일' in userText:
  #   app.slot_dict['types'] = '에일'
  
  # # 슬롯으로 잡지만 사실 슬롯에 해당하는 단어가 아닌 경우
  # if 'types' in inferred_tags[0] and userText not in types:
  #   app.slot_dict['types'] = None
  #   message = '네가 찾는 건 없네ㅠㅠ' + mType
  #   return message
  
  # elif 'abv' in inferred_tags[0] and userText not in abv:
  #   app.slot_dict['abv'] = None
  #   message = '네가 찾는 건 없네ㅠㅠ' + mAbv
  #   return message

  # elif 'taste' in inferred_tags[0] and userText not in taste:
  #   app.slot_dict['taste'] = None
  #   message = '네가 찾는 건 없네ㅠㅠ' + mTaste
  #   return message

  # elif 'flavor' in inferred_tags[0] and userText not in flavor:
  #   app.slot_dict['flavor'] = None
  #   message = '네가 찾는 건 없네ㅠㅠ' + mFlavor
  #   return message

  #options = {'beer_types':'종류', 'beer_abv':'도수', 'beer_flavor':'향', 'beer_taste':'맛'}
  empty_slot = [options[k] for k in app.slot_dict if not app.slot_dict[k]]
  filled_slot = [options[k] for k in app.slot_dict if app.slot_dict[k]]    
  print("empty_slot :", empty_slot)
  print("filled_slot :", filled_slot)

  # for k in slot_text:
  uni_li = list(set([slot_text[k] for k in slot_text]))
  print('uni_li :', uni_li)

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

  elif userText in endings:
    message = 'Okay bye...'
    return message

  # if uni_li[0] == '':
  # if ('종류' in empty_slot and '도수' in empty_slot and '향' in empty_slot and '맛' in empty_slot):
  #   message = '맥주의 종류, 도수, 향, 맛을 넣어서 다시 입력해주세요'   

  # if uni_li[0] == '': 태깅된 슬롯이 0개
  if ('종류' in empty_slot and '도수' in empty_slot and '향' in empty_slot and '맛' in empty_slot):
    message = '원하는 맥주의 종류는? 도수는? 향은? 맛은? 어떤 게 좋니??'
    #return message

  # 슬롯을 하나 이상 만족시키는 경우
  elif ('종류' in filled_slot or '도수' in filled_slot or '향' in filled_slot or '맛' in filled_slot):
    tmp_li = []
    for i in range(0, len(inferred_tags[0])):
        if not inferred_tags[0][i] == "O":
            tmp_li.append(slot_text[inferred_tags[0][i]])
  
    msg_li = list(set(tmp_li))
    message = chatbot_msg(msg_li)
    #return message
    print('tmp_li :', tmp_li)
            
    #if userText.strip().startswith("예"):
    if userText in yes:
      message = "어떤 걸 찾고 있어?"
    
    #elif userText.strip().startswith("아니오"):
    elif userText in no:
      # last_msg = "바로 맥주 추천해줄게"
      # return last_msg
      message = '라져! 네게 딱 맞을 맥주를 찾고 있는 중...'
      init_app(app)
      # return message
            
  return message

# def catch_slot(i, inferred_tags, text_arr, slot_text):
#   if not inferred_tags[0][i] == "O":
#     word_piece = re.sub("_", " ", text_arr[i])
#     slot_text[inferred_tags[0][i]] += word_piece

#inffered_tags = ['O', 'abv', 'abv', 'O', 'type', 'type', 'type', 'O', 'O', 'O', 'O', 'O', 'flavor', 'flavor', 'flavor', 'flavor', 'O', 'O']
#text_arr = ['나는_', '7', '도_', '넘는_', '흑', '맥', '주로_', '주', '문', '하고_', '싶', '어_', '스', '모', '키', '한_', '걸', '로_']
#slot_text = {'beer_abv': '7도', 'beer_flavor': '', 'beer_taste': '', 'beer_types': '흑맥주'}

def init_app(app):
    app.slot_dict = {'types': [], 'abv':[], 'flavor':[], 'taste':[]}

# 공백, 중복 제거 함수    
def chatbot_msg(msg_li):
    for k in app.slot_dict:
        msg_li.extend(app.slot_dict[k])
                
    for i in range(len(msg_li)):
        msg_li[i] = msg_li[i].strip()
    
    msg_li = list(set(msg_li))
    message = f"{msg_li} 말이지? <br />\n 추가적인 고려 사항이 있니?"
    
    # message = "너가 찾는 맥주 : {} <br />\n더 고려할 사항이 있니? (예 / 아니오)".format(msg_li)
    
    # ask_opt = True
    return message
