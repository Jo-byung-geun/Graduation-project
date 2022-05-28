import pandas as pd
from collections import Counter
from konlpy.tag import Komoran

csv = pd.read_csv('2020_1_교양.csv', encoding='cp949')

for k in range(len(csv)):

    # 강의개요 컬럼 리스트로 추출
    outline = csv['outline']
    outline = outline.values.tolist()
    # print(outline)

    # 강의목표 컬럼 리스트로 추출
    goal = csv['goal']
    goal = goal.values.tolist()
    # print(goal)

    # 강의개요와 강의목표 합쳐서 txt 파일에 쓰기
    outlineNgoal = open('input.txt', 'w')
    outlineNgoal.write(outline[k])
    outlineNgoal.write(goal[k])
    outlineNgoal.close()

    # txt 파일 읽어오기
    filename = "input.txt"
    f = open(filename, 'r', encoding='euc-kr')
    news = f.read()
    keyword = []
    result = ''

    if len(news) == 2:
        keyword = '없음'
        continue

    else:
        # Komoran객체 생성
        komoran = Komoran()
        noun = komoran.nouns(news)
        # print(noun)

        for i, v in enumerate(noun):
            if len(v) < 2:
                noun.pop(i)

        count = Counter(noun)
        f.close()

        # 명사 빈도 카운트
        noun_list = count.most_common(100)
        for v in noun_list:
            # 2번 이상 등장한 단어만 키워드로 저장
            if v[1] < 2:
                continue
            else:
                result += v[0] + ', ' + str(v[1]) + '/'

        print(result)
        #print(noun_list)
        #keyword = r


    # csv 특정 위치에 keyword를 쓰고 새로운 csv 파일 만들기
    csv.iat[k, 10] = result
    csv.to_csv('2020_1_교양_키워드.csv', index=False, encoding='euc-kr')


