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

    df.to_csv('celltrion_data.csv')  # csv저장

    price_data = pd.read_csv('celltrion_data.csv')  # csv저장 읽어오기
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
