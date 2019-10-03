# -*- coding: utf-8 -*-

# =============================================================================
# # BIZ&AI lab 6기 과제_4
# 
# 
# * train, validation, test set으로 나누기 & Word2Vec 작성 *
#  
# 
# 웹크롤링 데이터('hw_4_data.csv')를 활용하여 ANN, CNN, RNN 과제를 진행하려합니다.
# 
# 그에 앞서 필요한 과정을 과제로 드립니다.
# 
# 
# 1. train, validation, test set으로 나누기
# 
# 머신러닝 기법을 활용할때는
# 
# 데이터를 학습용(train), 검증용(validation), 테스트용(test) set으로 구분하여야 합니다.
#
# 수집한 데이터의 경우 데이터 균형이 맞지 않으므로 균형을 맞춰줄 필요가 있습니다.
# 
# 오버샘플링, 언더샘플링 등 각자 원하는 방법으로 긍부정 비율을 맞춰주시기 바랍니다.
# 
# 
# 긍정 데이터(label 1) 60% - trian, 20% - validation, 20% - test
# 부정 데이터(label 0) 60% - trian, 20% - validation, 20% - test
# 
# 위의 비율로 데이터를 train, validation, test set으로 나누어 저장하는 코드를 작성하시기 바랍니다.
# (이때 위에서 전처리한 데이터를 사용하시기 바랍니다.)
# 
# 
# 
# 2. Word2Vec 작성
# 
# 수집한 리뷰글 데이터를 활용하여 Word2Vec를 작성하시기 바랍니다.
# (원래는 train 데이터만으로 Word2Vec를 만들어야 하나 일단 수집한 모든 데이터를 통해 작성)
# 
# 강의때는 명사만을 가지고 만들어서 동사, 형용사 등 필요한 태그에 대한 Word2Vec를 만들지 못했습니다.
# 
# 
# 개인이 생각하는 필요하다고 생각하는 태그를 가지고 Word2Vec을 작성하시기바랍니다.
# 
# 또한 하이퍼파라미터도 각자 결정하시기 바랍니다.
# 
# 
# 
# 
# 8/? 까지 1.작성한 코드, 2.과제 발표할 PPT 를 메일로 보내주시기 바랍니다.
# 
# =============================================================================

from sklearn.model_selection import train_test_split # 데이터를 랜덤하게 train, validation, test로 나누는 것. 
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import os



def fn_data_split(data_df, train_file_name, vali_file_name, test_file_name):
    pos = data_df.loc[data_df['label']==1,:]
    neg = data_df.loc[data_df['label']==0,:]
    neg_c = neg.copy()
    for i in range(6): # 오버샘플링 
        neg = pd.concat((neg, neg_c), axis=0, ignore_index=True) 
    neg = pd.concat((neg,neg.loc[:3968,]))
    pos_train, pos_tmp, neg_train, neg_tmp = train_test_split(pos, neg, train_size=0.6)
    pos_val, pos_test, neg_val, neg_test = train_test_split(pos_tmp, neg_tmp, train_size=0.5)
    train_set = pos_train.append(neg_train)
    val_set = pos_val.append(neg_val)
    test_set = pos_test.append(neg_test)
    
    train_set.to_csv(train_file_name, header=True, index=False, encoding='cp949')
    val_set.to_csv(vali_file_name, header=True, index=False, encoding='cp949')
    test_set.to_csv(test_file_name, header=True, index=False, encoding='cp949')
    
def fn_create_word2vec(data_df, w2v_model_name):
    okt = Okt()
    result=[]
    pos = ['Verb', 'Adjective', 'Noun','Josa', 'Eomi','Conjunction','Adverb','Determiner']
    for x in range(len(data_df)):
        words = okt.pos(data_df.loc[x,'review'])
        tmp_result=[]
        for y,z in words:
            if z in pos: tmp_result.append(y)
        result.append(tmp_result)
        
    w2v_model = Word2Vec(result, size=100, window=5, min_count=5, sg=0)
    w2v_model.save(w2v_model_name)       
    
    return w2v_model


if __name__ == "__main__":
    train_file_name = 'hw_4_train_data_CYK.csv'
    vali_file_name = 'hw_4_validation_data_CYK.csv'
    test_file_name = 'hw_4_test_data_CYK.csv'
    
    w2v_model_name = 'hw_4_word2vec_CYK.model'
    
    os.chdir(r'C:\Users\CHOYEONGKYU\Desktop\프학\4주차')
    data_file_name = 'hw_4_data.csv'
    data_df = pd.read_csv(data_file_name, encoding='cp949')[['label', 'review']]
    print('data_df shape - ', data_df.shape)
    
    data_df = data_df.drop_duplicates()
    print('data_df(drop_duplicates) shape - ', data_df.shape)
    
    print(data_df.groupby(['label'])['label'].count())

# =============================================================================

    os.chdir(r'C:\Users\CHOYEONGKYU\Desktop\프학 과제\4주차')
    fn_data_split(data_df, train_file_name, vali_file_name, test_file_name)
    w2v_model = fn_create_word2vec(data_df, w2v_model_name)
    w2v_model = Word2Vec.load(w2v_model_name)
    
    for test_word in ['월요일' , '배송', '빠르다', '좋다', '감사', '별로']:
        print('*'*50 + '\n' + test_word)
        pprint(w2v_model.wv.most_similar(test_word, topn=5))

            

'''

**************************************************
월요일
[('금요일', 0.959784746170044),
 ('목요일', 0.956427276134491),
 ('수요일', 0.9455667734146118),
 ('화요일', 0.9429277777671814),
 ('토욜', 0.9383894801139832)]
**************************************************
배송
[('파른', 0.7278835773468018),
 ('리오네', 0.7141883373260498),
 ('송도', 0.7126674652099609),
 ('명절', 0.6764638423919678),
 ('송이', 0.6738250255584717)]
**************************************************
빠르다
[('감솨', 0.7879383563995361),
 ('파른', 0.7672903537750244),
 ('총알', 0.7666589021682739),
 ('빨랏', 0.7610298991203308),
 ('빨르다', 0.7538026571273804)]
**************************************************
좋다
[('잘삿어', 0.7105856537818909),
 ('정말로', 0.7042036056518555),
 ('욧', 0.7030841708183289),
 ('아영', 0.7001270055770874),
 ('좋아욤', 0.6972665786743164)]
**************************************************
감사
[('하비다', 0.7892617583274841),
 ('감솨', 0.7831730842590332),
 ('고맙다', 0.7772567272186279),
 ('감사하다', 0.7344866394996643),
 ('힙니', 0.7170889973640442)]
**************************************************
별로
[('별루', 0.6891950368881226),
 ('그닥', 0.6276479959487915),
 ('역다', 0.5523307919502258),
 ('이외', 0.5468701124191284),
 ('안좋다', 0.5464729070663452)]

'''
