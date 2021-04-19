from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pyperclip

driver = webdriver.Chrome('C:\chromedriver.exe')
driver.get('https://www.naver.com/')
time.sleep(1)

# 로그인 버튼을 찾고 클릭합니다
login_btn = driver.find_element_by_class_name('link_login')
login_btn.click()
time.sleep(1)

# id, pw 입력할 곳을 찾습니다.
tag_id = driver.find_element_by_name('id')
tag_pw = driver.find_element_by_name('pw')
tag_id.clear()
time.sleep(1)

# id 입력
tag_id.click()
pyperclip.copy('dksthaud')
tag_id.send_keys(Keys.CONTROL, 'v')
time.sleep(1)

# pw 입력
tag_pw.click()
pyperclip.copy('Thtnlsla3913@')
tag_pw.send_keys(Keys.CONTROL, 'v')
time.sleep(1)

# 로그인 버튼을 클릭합니다
login_btn = driver.find_element_by_id('log.login')
login_btn.click()

from bs4 import BeautifulSoup

driver.get('https://mail.naver.com/')
html=driver.page_source
soup=BeautifulSoup(html,'lxml')

#메일 제목
title_list=soup.find_all('strong','mail_title')
time.sleep(2)
#메일 클릭
mail_text = driver.find_element_by_xpath('//*[@id="list_for_view"]/ol/li[1]/div/div[2]/a[1]/span[1]')
mail_text.click()
time.sleep(2)
#메일 삭제
test = driver.find_element_by_xpath('//*[@id="readBtnMenu"]/div[1]/span[2]/button[2]')
test.click()
time.sleep(2)
#모든 메일 출력
# for title in title_list:
#     print(title.text)
