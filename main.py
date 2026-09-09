#주어진 txt파일을 파싱하여, 
#https://drive.google.com/drive/folders/145_yo6nbFdvqvCqvczGDGJ5CZCOZsY5g
#main.py를 실행했을 때 각 리뷰에 대한 리플이 자동으로 생성되도록 하시오.
import pandas as pd 
#1. pandas 라이브러리를 사용해서 txt파일을 읽어오기
def load_reviews(path):
    df = pd.read_csv(path, sep='\t')
    #print(df.columns)
    #print(df['comment'])
    return df

import templates as T
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
def build_reply_chain(chat):

    #from_template, from_message
    prompt = ChatPromptTemplate.from_template(T.REPLY_TEMPLATE)

    #return 프롬프트 | 챗 | 파서 -> Str, Json, Structured
    return prompt | chat | StrOutputParser()

#리뷰를 한 줄 한 줄 읽어서 build_reply_chain에 넣어주는 함수
def generate_reply(reviews, chat):
    reply_chain = build_reply_chain(chat)

    for i in range(len(reviews)):
        # print()
        # print(review['label'])
        # print(type(review['label']))
        # #템플릿에서 뽑아올 2가지 요소
        sentiment = '긍정' if reviews.loc[i]['label'] == 1 else "부정"
        comment = reviews.loc[i]['comment']

        result = reply_chain.invoke({'sentiment' : sentiment, 'comment' : comment})

        print(f'[답글 생성] 손님 댓글 {comment}\n 사장님 댓글 {result}')

       
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
#비밀 키를 가져오는 역할을 함 -> .env
load_dotenv()
if __name__ == '__main__':
    #1. pandas 라이브러리를 사용해서 txt파일을 읽어오기
    df = load_reviews('./tarr_train.txt')

    #temperature(0~1) : 창의성
    chat = ChatOpenAI(temperature = 0.7, model = 'gpt-4o')

    generate_reply(df, chat)


    #2. 오늘 한 chain 함수를 이용해서 댓글 분류(선택) / 긍정-부정
    #3. 오늘 한 chain 함수를 이용해서 댓글에 대한 답글 생성(필수) > '페르소나' 부여 가능