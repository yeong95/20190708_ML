# -*- coding: utf-8 -*-

# =============================================================================
# # BIZ&AI lab 6기 과제_1
# 
# 
# * 문자열 암호화 / 복호화 함수 만들기 *
# 
# 
# 모든 문자는 숫자로된 아스키코드로 이루어져있습니다.
# 
# 이를 활용하여 문자열을 암호화하고 다시 복호화하는 함수를 만들시기 바랍니다.
# 
# 
# python 내장함수인 ord() 함수는 문자의 유니코드 값을 반환합니다.
# (예를 들어, ord('가') 를 입력하면 해당하는 유니코드 값 44032 를 출력합니다.)
# 
# 이와 반대로 chr() 함수는 유니코드 값을 문자로 반환합니다.
# (예를 들어, chr(44032) 를 입력하면 해당하는 문자 '가' 를 출력합니다.)
# 
# 
# 
# 1. 문자열 암호화 함수(encryption_fn) 만들기
# - 암호화할 문자열의 문자를 아스키코드 값으로 변환한다.
# - 유니코드 값으로 변형시킨다.
#   10~10,000의 자리 수를 1~1,000의 자리로 바꾸고 1의 자리 수를 10,000의자리로 바꾼다.
#   (예를 들어, ord('가') : 44032 -> 24403 으로 변형,  ord('a') : 97 -> 70009 로 변형)
# - 이를 다시 문자열로 변형하여 return 한다.
# 
# 2. 문자열 복호화 함수(decryption_fn) 만들기
# - 복호화할 문장을 각 문자별로 유니코드 값으로 변환한다.
# - 유니코드 값을 되돌린다.
#   1~1,000의 자리 수를 10~10,000의 자리로 바꾸고 10,000의 자리 수를 1의자리로 바꾼다.
# - 이를 다시 문자열로 변형하여 return 한다.
# 
# 
# 아래 코드의 함수 부분을 완성하여 제출하시기 바랍니다.
#
# 7/17 까지 1.작성한 코드, 2.과제 발표할 PPT 를 메일로 보내주시기 바랍니다.
#
# =============================================================================


tex=[]
number=[]

def encryption_fn(text):
    for v in text:
        tex.append(ord(v)) 
    def change_num(tex):
        for h in tex:
            first_digit=(h%10)*10000
            rest=h//10
            final=first_digit+rest
            number.append(final)
    change_num(tex)
    enc_text=[]
    for m in number:
        enc_text.append(chr(m))
    return enc_text

tex2=[]
number2=[]
def decryption_fn(text):
    for k in enc_text:
        tex2.append(ord(k))
    def change_num(tex2):
        for m in tex2:
            last_digit=m//10000
            rest=(m-last_digit*10000)*10
            final=last_digit+rest
            number2.append(final)
    change_num(tex2)
    denc_text=[]
    for n in number2:
        denc_text.append(chr(n))
    return denc_text


text = 'HW_1. 암호화 - 복호화'

enc_text = encryption_fn(text)
print('encryption : ', enc_text)
# encryption :  丧𑅸썙𕾔丣︛ᕬ𔷮丣썔丣蠕ᕬ𔷮

denc_text = decryption_fn(enc_text)
print('decryption : ', denc_text)
# decryption :  HW_1. 암호화 - 복호화
