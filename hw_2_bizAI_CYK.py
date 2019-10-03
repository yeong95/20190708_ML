# -*- coding: utf-8 -*-

# =============================================================================
# # BIZ&AI lab 6기 과제_2
# 
#
# * 로그 데이터 전처리 *
#  
#  
# 1. 파일 및 변수 설명
# 1-1. hw_2_label_data.csv
#     : 개인의 아이디, 성별, 나이, 나이대를 정리한 파일
#      - CUS_ID : 개인 아이디 (숫자 값)
#      - GENDER : 성별 (M F 범주 값)
#      - AGE : 나이 (숫자 값)
#      - AGE_CLASS : 나이대 (범주 값) 
# 
# 1-2. hw_2_raw_data.csv
#     : 개인의 인터넷 사용기록(log data)
#      - CUS_ID : 개인 아이디 (숫자 값)
#      - TIME_ID : 인터넷 사용 시간 (년월일시간 값)
#                ex) 2012083016 -> 2012년 8월 30일 16시
#      - ACT_NM : 접속 사이트 종류 (188가지의 범주 값)
# 
# 1-3. hw_2_ex_data.csv
#     : 데이터 전처리 후 파일 (예시)
#         
# 1-4. hw_2_pickle_data_col.pkl  <-컬럼 네임을 저장해 놓은 것. 
#     : GENDER_col_list, DAY_col_list, TIME_col_list, ACT_col_list 값이 저장된 pickle 파일
# 
# 1-4-1. GENDER_col_list
#     : 저장할 데이터의 성별 부분 컬럼명 
#         ['GENDER']
# 
# 1-4-2. DAY_col_list
#     : 저장할 데이터의 요일 부분 컬럼명 
#         ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
# 
# 1-4-3. TIME_col_list
#     : 저장할 데이터의 시간 부분 컬럼명 
#         ['0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', '11h', ...... ]
# 
# 1-4-4. ACT_col_list
#     : 저장할 데이터의 사이트 종류 부분 컬럼명 
#         ['B2B', 'IT/통신', 'NGO', 'PC게임', 'PC방/게임방', 'SaaS/ASP', '가격비교', '가구/인테리어 쇼핑몰', ...... ]
# 
# 
# 2. 과제
# 2-1. TIME_ID 를 사용하여 각 인터넷 사용 기록의 시간과 요일에 대한 열을 만든다.
#   (datetime 라이브러리를 사용하면 요일을 알아낼 수 있음. 다른 라이브러리를 사용해도 됨.)
#    ex)
#     CUS_ID    TIME_ID       ACT_NM     TIME_HOUR     DAY
#       9999   2012070412      일간지       12h         WED
#       9999   2012072414       검색        14h         TUE
#       9999   2012081613       검색        13h         THU
#       9999   2012092512       검색        12h         TUE
#       9999   2012101215       포털        15h         FRI
#     
# 2-2. 개인별로 요일당, 시간당, 사이트 종류당 접속 횟수를 카운트한다.
#    ex)
#     9999 ->
#      요일 :  WED    1
#             TUE    2
#             THU    1
#             FRI    1
#     
#      시간 :  12h    2
#             14h    1
#             13h    1
#             15h    1
#     
#      종류 :  일간지    1[]
#             검색    3
#             포털    1
# 
# 2-3. 위의 데이터에 개인별 총 접속횟수로 나누고, 성별 데이터를 포함하여 DataFrame 형식으로 save_df를 생성한다.
#    ex)
#     9999 ->
#         GENDER                   F
#         MON                      0
#         TUE                      0.4
#         WED                      0.2
#         THU                      0.2
#         FRI                      0.2
#         SAT                      0
#         SUN                      0
#         
#         0h                       0
#         1h                       0
#         2h                       0
#         3h                       0
#         4h                       0
#         5h                       0
#         6h                       0
#         7h                       0
#         8h                       0
#         9h                       0
#        10h                       0
#        11h                       0
#        12h                       0.4
#        13h                       0.2
#        14h                       0.2
#        15h                       0.2
#        16h                       0
#        17h                       0
#        18h                       0
#        19h                       0
#        20h                       0
#        21h                       0
#        22h                       0
#        23h                       0
#        
#        일간지                    0.2
#        검색                      0.6
#        포털                      0.2
#        학술정보                  0
#        취업                      0
#        ......(생략)     
# 
# 2-4. 생성한 save_df를 'hw_2_data_ooo.csv'로 저장한다.
#       'hw_2_ex_data.csv'처럼 데이터 저장
# 
# 
# 
# 아래 코드 함수의 save_df를 작성하는 부분을 완성하시기 바랍니다.
# 피클 파일: 데이터가 날아가지 않도록 저장해둔것. 
# 
# 
# 7/24 까지 1.작성한 코드, 2.생성한 데이터 파일, 3.과제 발표할 PPT 를 메일로 보내주시기 바랍니다.
# =============================================================================
 


import pandas as pd
import numpy as np
import datetime
import pickle
import os


data_file_path = r'C:\Users\CHOYEONGKYU\Desktop\프학\2주차'
os.chdir(data_file_path)

with open('hw_2_pickle_data_col.pkl', 'rb') as f:
    data_col = pickle.load(f)
    
GENDER_col_list, DAY_col_list, TIME_col_list, ACT_col_list = data_col

column_list = ['CUS_ID'] + GENDER_col_list + DAY_col_list + TIME_col_list + ACT_col_list


def print_whichday(year, month, day) : #요일 구해주는 함수
    r = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    aday = datetime.date(year, month, day)
    bday = aday.weekday()
    return r[bday]

def data_preprocessing(raw_data_file, label_data_file, save_data_file):
    raw_data = pd.read_csv(raw_data_file, encoding='cp949')
    label_data = pd.read_csv(label_data_file)
    
    Time=[]
    year=[]
    month=[]
    day=[]
    for x in raw_data['TIME_ID']:  #시간, 연도, 월, 일로 나눠주는 작업. 
        Time.append((str(x)[8:])+'h')
        year.append((str(x)[0:4]))
        month.append((str(x)[4:6]))
        day.append((str(x)[6:8]))
    
    i=0
    for x in Time:
        if int(x[0:2])<10: Time[i]=x[1:]
        i=i+1
        
    raw_data['TIME_HOUR']= pd.DataFrame(Time) #시간을 구한 것. 
    
    time_df=pd.DataFrame({'year':year,'month':month,'day':day}) # 요일 구한 것. 
    DAY=[]
    for x in range(len(time_df)):
        DAY.append(print_whichday(int(time_df['year'][x]),int(time_df['month'][x]),int(time_df['day'][x])))
    raw_data['DAY']=pd.DataFrame(DAY)
    
    day_count=raw_data.groupby(['CUS_ID','DAY'])['DAY'].count()
    time_count=raw_data.groupby(['CUS_ID','TIME_HOUR'])['TIME_HOUR'].count()
    act_count=raw_data.groupby(['CUS_ID','ACT_NM'])['ACT_NM'].count()    
    
    frames=[]
    for x in day_count.index.levels[0]:
        A=day_count[x]/day_count[x].sum()
        B=time_count[x]/time_count[x].sum()
        C=act_count[x]/time_count[x].sum()
        D=pd.DataFrame({'CUS_ID':x,'GENDER':label_data.ix[label_data['CUS_ID']==x,1].values})
        E=pd.DataFrame(A.append(B).append(C)) #시리즈 이어 붙이고 데이터 프레임 만들기
        df_con=pd.concat((D,E.T),axis=1)
        frames.append(df_con)
    
    result=pd.concat(frames,sort=False,join='outer') #모든 데이터프레임 합쳐주기. 
    result.index=range(len(result))
    result.replace(np.nan,0,inplace=True)   
   
     
    save_df = pd.DataFrame(result, columns=column_list)
    
# =============================================================================
#     save_df를 작성하시기 바랍니다.
# =============================================================================
    
    save_df = save_df[column_list]
    save_df.to_csv(save_data_file, header=True,index=False,encoding='cp949')
    

if __name__ == "__main__":
    data_preprocessing('hw_2_raw_data.csv', 'hw_2_label_data.csv', 'hw_2_data_ooo.csv')
