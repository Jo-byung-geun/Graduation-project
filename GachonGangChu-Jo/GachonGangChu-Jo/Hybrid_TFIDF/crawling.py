from selenium import webdriver
from selenium.webdriver.support.ui import Select
import pandas as pd

driver = webdriver.Chrome('C:/Users/USer/chromedriver_win32/chromedriver.exe')

# webdriver가 페이지에 접속하도록 명령
driver.get('http://sg.gachon.ac.kr/main?attribute=timeTable&gbn=P&lang=ko%27%27')
driver.maximize_window()

# 검색할 년도 설정
select = Select(driver.find_element_by_id('year'))
select.select_by_value('2020')

# 검색할 학기 설정
select = Select(driver.find_element_by_id('hakgi'))
select.select_by_value('20')

# 전공, 교양 선택
select = Select(driver.find_element_by_id('p_isu_cd'))
select.select_by_value('1')

# 단대 선택 (option에 value가 없어서 값을 찾는?형식으로 설정)
ddelement = Select(driver.find_element_by_id('p_univ_cd'))
ddelement.select_by_visible_text('IT융합대학')

# 과 선택 (마찬가지로 값을 찾는 형식으로 설정)
ddelement = Select(driver.find_element_by_id('p_maj_cd'))
ddelement.select_by_visible_text('AI·소프트웨어학부(소프트웨어전공)')

# 조회버튼 클릭
button = driver.find_element_by_css_selector('button').click()

GU_dict = {'outline':[],
           'goal' : []
            }

# 강의계획서 아이콘 클릭
for i in range(0,999):
    plan = driver.find_elements_by_css_selector('.xi-paper')[i].click()

    #창 바꾸기
    driver.switch_to_window(driver.window_handles[-1])

    table = driver.find_elements_by_class_name('tbl-view')[2]
    tbody = table.find_element_by_tag_name("tbody")
    rows = tbody.find_element_by_tag_name("tr")
    body= rows.find_element_by_tag_name("td")

    GU_dict['outline'].append(body.text)

    table = driver.find_elements_by_class_name('tbl-view')[3]
    tbody = table.find_element_by_tag_name("tbody")
    rows = tbody.find_element_by_tag_name("tr")
    body= rows.find_element_by_tag_name("td")

    GU_dict['goal'].append(body.text)

    driver.close()
    driver.switch_to.window(driver.window_handles[0])

test = pd.DataFrame.from_dict(GU_dict)
test.tail()