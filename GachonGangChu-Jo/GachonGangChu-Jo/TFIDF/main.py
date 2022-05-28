# -*- coding: utf-8 -*-
import pandas as pd # 데이터프레임 사용을 위해
from sklearn.metrics.pairwise import cosine_similarity

# 키워드 csv파일 가져오기
csv = pd.read_csv('2020_1_교양_키워드.csv', encoding='cp949')

# 과목명 리스트형태로 가져오기
course_name_list = csv['교과명']
course_name_list = course_name_list.values.tolist()
# print(course_name_list)

# 키워드 가져오기
keyword1 = csv['keyword']
keyword_Word = []
for k in range(len(csv)):

    # 키워드가 없는 경우 (강의 개요, 강의 목표 없어서 키워드가 없음)
    if keyword1[k] is 'x':
        #print('키워드 없음')
        continue

    else:
        keywordList = list(keyword1[k].split('/'))

        for i in range(len(keywordList)):
            # 한 강의의 키워드 끝을 나타내는 '' 가 나오면 다음으로 넘어가기
            if keywordList[i].split(',')[0] == '':
                # print('끝')
                continue
            # 이미 키워드 사전에 있는 키워드라면 건너 뛰기
            elif keywordList[i].split(',')[0] in keyword_Word:
                # print('똑같은거')
                continue

            else:
                # print(keywordList[i].split(',')[0])
                keyword_Word.append(keywordList[i].split(',')[0])

    #print(keywordList)
#print()
#print(keyword_Word)


# 데이터프레임 만들기
dataFrame = pd.DataFrame(index=course_name_list, columns=keyword_Word)

for k in range(len(csv)):
    course_name = csv.iat[k, 2]
    #print(course_name)

    # 키워드가 없는 경우 (강의 개요, 강의 목표 없어서 키워드가 없음)
    if csv.iat[k, 10] is 'X':
        #print('키워드 없음')
        continue

    elif csv.iat[k, 10] is not 'X':
        keywordList = list(keyword1[k].split('/'))
        for j in range(len(keywordList) - 1):
            #print(keywordList[j].split(', '))
            #print(keywordList[j].split(', ')[0])  # 키워드 한글
            #print(keywordList[j].split(', ')[1])  # 숫자
            dataFrame.loc[course_name, keywordList[j].split(', ')[0]] = keywordList[j].split(', ')[1]

#display(dataFrame)

# 비어있는 칸 0으로 채우기
dataFrame.fillna(0, inplace=True)
# 데이터프레임 csv 파일로 저장
dataFrame.to_csv('TFIDF_교양.csv', index=True, encoding='euc-kr')

# 데이터프레임 어레이로 바꾸기
# 이거 필요없나봐ㅋㅎ
# dataFrame_array = dataFrame.to_numpy()

# 코사인 유사도 계산하기
cosine_matrix = cosine_similarity(dataFrame, dataFrame)
#print(cosine_matrix)
#print()

# 코사인 유사도 계산한거 csv 파일로 저장하기 (dataframe으로 다시 바꿔서 csv로 저장)
cosine_matrix_df = pd.DataFrame(cosine_matrix)
cosine_matrix_df.to_csv('cosine_교양.csv', index=True, encoding='euc-kr')


course_to_index = dict(zip(csv['교과명'], csv.index))
# print(course_to_index)

def get_recommendations(course, cosine_matrix=cosine_matrix):
    # 사용자가 입력한? 강의 이름의 인덱스를 받아온다.
    idx = course_to_index[course]

    # 해당 강의와 모든 강의와의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_matrix[idx]))

    # 유사도에 따라 강의들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 강의를 받아온다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 강의의 인덱스를 얻는다.
    movie_indices = [idx[0] for idx in sim_scores]

    # 가장 유사한 10개의 강의의 제목을 리턴한다.
    return csv['교과명'].iloc[movie_indices]

# 사용자가 입력하는걸로 바꾸기
print("수강했던 과목과 비슷한 과목을 추천해드립니다.")
course_name = input("수강했던 과목을 입력하세요: ")
print()
print(get_recommendations(course_name))

