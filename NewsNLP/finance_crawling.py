import pandas_datareader as wb
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup


def finance():
    pd.set_option('precision', 4)

    start = datetime.datetime(2019, 8, 1)  # 데이터 저장 시작날짜
    end = datetime.datetime(2020, 2, 1)  # 데이터 저장 끝나는 날짜
    df_null = wb.DataReader("068270.KS", "yahoo", start, end)  # 데이터읽어오기 yfinance종목코드 입력
    # 현대차 005389.KS
    # LG전자 066570.KS
    # 한국조선해양 009540.KS
    # 셀트리온 068270.KS
    # 서울반도체 046890.KQ
    df = df_null.dropna()  # 결측치 제거

    # (Price : 당일 대비 다음날 주가가 상승했으면 1, 하락했으면 0표시)
    df['Price'] = 0
    for i in range(0, 120):
        if df['Close'][i] < df['Close'][i + 1]:  # 다음날 종가가 당일 종가 클 때
            df['Price'][i] = 1  # 상승 시 1
        else:
            df['Price'][i] = 0  # 하락 시 0

    df.to_csv('ship_data.csv')  # csv저장

    price_data = pd.read_csv('ship_data.csv')  # csv저장 읽어오기
    df_0 = price_data[price_data['Price'] == 0]['Date']
    date_0 = []
    for i in range(0, len(df_0)):
        date_0.append(str(df_0.tolist()[i])[:10].replace('-', '.'))  ##Price 1인 날짜에 -없애고 공백으로 대체

    df_1 = price_data[price_data['Price'] == 1]['Date']
    print(date_0)
    date_1 = []
    for i in range(0, len(df_1)):
        date_1.append(str(df_1.tolist()[i])[:10].replace('-', '.'))  # Price 1인 날짜에 -없애고 공백으로 대체

    print(date_1)



finance()

import requests
import time
from bs4 import BeautifulSoup
import pandas as pd

page_list = []
title_list = []
time_list = []


def car_data():
    price_data = pd.read_csv('ship_data.csv')

    return list(price_data['Price']), list(price_data['Date'])


def save_csv(text, time):  # csv파일 생성
    data, dates = car_data()
    title_list.append(text)  # .append csv파일 저장할때 같이쓰면 안됨
    time_list.append(time.replace('.', '-', 2).replace('.', ''))  # replace 중첩해서 날짜를 -로 바꾸고 마지막 .제거
    title_df_2 = pd.DataFrame(title_list, columns=['title'])
    title_df_2['day'] = time_list  # 날짜에 .표시 제거

    is_matched = False
    prices = []
    #ship_data.csv의 날짜와 뉴스의 날짜가 같을 시 ship_data.csv의 price가 change에 저장
    for b in time_list:
        for i, a in enumerate(dates):
            if a == b:
                price = data[i]
                is_matched = True

        if is_matched:
            prices.append(price)
        else:
            prices.append(None)
        is_matched = False

    title_df_2['change'] = prices
    print(title_df_2)
    title_df_2 = pd.concat([title_df_2])
    title_df_2.to_csv('new_title.csv', index=True, encoding='utf-8-sig')


for page in range(200):
    page_list.append(
        "https://search.naver.com/search.naver?&where=news&query=%EB%8C%80%EC%9A%B0%EC%A1%B0%EC%84%A0%ED%95%B4%EC%96%91&sm=tab_pge&sort=1&photo=0&field=0&reporter_article=&pd=3&ds=2019.04.15&de=2019.04.30&docid=&nso=so:dd,p:from20190401to20190415,a:all&mynews=0&start={}&refresh_start=0".format(
            str(page * 10 + 1)))  # 네이버 뉴스 기사 페이지 넘기기
while True:
    i = 1
    for page in page_list:
        res = requests.get(page, headers={'User-Agent': 'Mozilla/5.0'})
        res.raise_for_status()

        html = BeautifulSoup(res.text, 'html.parser')
        new_title = html.find_all("a", attrs={"class": "news_tit"})  # 뉴스 제목 크롤링
        time = html.find_all("span", attrs={"class": "info"})  # 날짜 출력
        end_croling = html.find("a", attrs={"class": "btn_next"})  # 크롤링 종료를 위한 버튼 찾기

        print(len(new_title), len(time))
        for n, t in zip(new_title, time):  # 변수 두개 for으로 쓰기
            print("날짜 = {} 기사제목 = {}".format(t.text, n.text))
            save_csv(n.text, t.text)

        print("*" * 30)
        print(i, "페이지")
        i += 1
        if end_croling.get("aria-disabled") == "true":  # 다음 페이지로 넘어가는 이벤트가 없으면 정지
            break  # for문을 끝내기 위한 if문

    if end_croling.get("aria-disabled") == "true":  # true면 마지막페이지 false면 활성화
        break  # while문을 끝내기 위한 if문