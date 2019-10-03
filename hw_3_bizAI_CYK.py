#*- coding: utf - 8 -*-

# =============================================================================
# # BIZ&AI lab 6기 과제_3
# 
# 
# * 11번가 제품 리뷰글 및 별점 크롤링 *
# 
# 
# 웹 크롤링 url 예시 :
# http://deal.11st.co.kr/product/SellerProductDetail.tmall?method=getProductReviewList&prdNo=1815000878&page=1
# 
# 
# 메인 함수에 있는 제품 아이디를 통해 리뷰글 및 별점을 수집하시기 바랍니다.
# 
# 파일을 저장할때는 
# 
# 'category', 'item_id', 'page(페이지 수)', 'score(별점)', 'sentiment(긍정 중립 부정)', 'review(리뷰 글)'
# 
# 위의 6개의 필드로 저장하시기바랍니다.
# 
# 
# 긍부정 감정은 
# 
# 별점 4, 5 점 -> 1
# 별점 3 점 -> 0
# 별점 1, 2 점 -> -1
# 
# 입니다.
# 
# 
# 메인 함수를 보시면 "크롤링 테스트용"이 있습니다.
# 
# 연습할때는 "크롤링 테스트용"(리뷰 70여건) 제품을 가지고 하시고
# 
# 과제는 "크롤링 과제용"(리뷰 10만여건)으로 하시기 바랍니다. (코드 실행시간 약 30분)
# 
# 
# 크롤링하는 페이지의 데이터가 이상이 있어서 생성한 코드에 따라 다르겠지만 에러가 발생할 수도 있습니다.
#
# 에러 발생시 해당 페이지를 직접 가서 확인해보고 try문을 활용하여 문제를 해결하시기 바랍니다.
#
# (UnicodeEncodeError 발생하는 리뷰는 건너 뛰기 바랍니다.)
# 
# 
# 아래 코드의 함수 부분을 완성하여 제출하시기 바랍니다.
# 
# 
# 7/31 까지 1.작성한 코드, 2.과제 발표할 PPT 를 메일로 보내주시기 바랍니다.
# 
# =============================================================================


from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re, time, os, csv



def fn_webcrawler(cw, category, item_id, page=0):
    global crawling_count
    while 1:
        page += 1
        url = 'http://deal.11st.co.kr/product/SellerProductDetail.tmall?method=getProductReviewList&prdNo={}&page={}'.format(item_id, page)
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        
        soup_break = soup.find_all(string='아직 작성된 리뷰가 없습니다.')
        if len(soup_break)>0 : break
        
        
        star_list=[]
        sentiment_list=[]        
        for soup_star in soup.find_all('span', class_=('selr_star')): #별점과 감정 구하기
            star_text = soup_star.text
            z = re.search(r'\s{1}\d{1}개{1}', star_text)
            star_number = z.string[z.start()+1:z.end()-1]
            if int(star_number)>=4:
                sentiment_list.append(1)
            elif int(star_number)==3:
                sentiment_list.append(0)
            elif int(star_number)<=2:
                sentiment_list.append(-1)
            star_list.append(star_number)
        
        review_list=[]
        for soup_review in soup.find_all('a', href='#this', id='reviewContTxt'):
                                         soup_text = soup_review.text.strip()
                                         review_list.append(soup_text)
                                         
        for x in range(len(star_list)):
             cw.writerow([ category , item_id , page , star_list[x] ,sentiment_list[x], review_list[x]])
            
                                
        # =============================================================================
        #         fn_webcrawler 함수를 완성하시오. 
        # =============================================================================
            
        time.sleep(0.01)
        
     


if __name__ == "__main__":
    start_time = time.time()
    
    crawling_count = 0
    
    save_file_path = r'C:\Users\CHOYEONGKYU\Desktop'
    os.chdir(save_file_path)
    save_file_name = 'hw_3_webcrawling_조영규.csv'
    
##     크롤링 테스트용
    #item_list = [('chair', 1815000878)]
##     크롤링 과제용
    item_list = [('chair', 1815000878), ('chair', 12423596), ('chair', 87595509), ('chair', 10843324), ('chair', 218190216), ('chair', 12623125) ,('chair', 50942984), ('char', 374683075)]
    #위의 거는 상품과 상품코드를 나타내는 것. 
    with open(save_file_name, 'w', newline='', encoding='cp949') as f:
        cw = csv.writer(f)
        cw.writerow(['category', 'item_id', 'page', 'score', 'sentiment', 'review'])
        
        for i, x in enumerate(item_list):
            category, item_id = x
            print('{0}\n{1} / {2}\n{3} - {4}\n{0}'.format('-'*100, i+1, len(item_list), category, item_id))
            try:
                fn_webcrawler(cw, category, item_id)
            except UnicodeEncodeError: pass
                
            tmp_running_time = time.time() - start_time
            print('%s\ntmp running time : %d m  %0.2f s\n%s\n'%('#'*100, tmp_running_time//60, tmp_running_time%60, '#'*100))
    
    running_time = time.time() - start_time
    print('%s\ntotal running time : %d m  %0.2f s\n%s'%('#'*100, running_time//60, running_time%60, '#'*100))
            
    crawling_data = pd.read_csv(save_file_name, encoding='cp949')
    print(crawling_data.shape)
    print(crawling_data.head())