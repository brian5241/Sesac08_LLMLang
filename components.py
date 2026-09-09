#LangChain의 기본 대화를 구성하는 3가지 요소
#1. 프롬프트와 파서 - 대화의 양식을 규정
#2. 체인 -> |(파이프), LCEL문법
#3. 메모리 -> 사용자와의 이전 대화 내용을 기억

#대화 대상 세팅
from langchain_openai import ChatOpenAI
from openai import OpenAI

#ChatPromptTemplate -> gpt나 11m에 질문할 내용을 정리하는 것
from langchain_core.prompts import ChatPromptTemplate
#StrOutputParser -> output으로 나온 결과물을 정제
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
#비밀 키를 가져오는 역할을 함 -> .env
load_dotenv()

#내가 사용할 템플릿
import templates as T

#내가 사용할 gpt모델
MODEL_NAME = 'gpt-4o'



def get_chat(temperature = 0.5, model = MODEL_NAME):
    return ChatOpenAI(temperature = temperature, model = model)

#프롬프팅 양식을 세팅
#매개변수 chat은 gpt와의 채팅방
#prompt | chat (미리 정해놓은 prompt를 chat채팅방에 넘겨주는 작동)
#chat | StrOutputParser() (chat의 답변을 StrOutputParser로 넘겨주는 작동)
def build_style_chain(chat):

    prompt = ChatPromptTemplate.from_template(T.STYLE_TEMPLATE)
    #LCEL 문법 => prompt를 chat에 넘겨주고 -> 그 결과를 strOutputParser에 다시 넣어주는 연결
    return prompt | chat | StrOutputParser()

#프롬프팅, 파서
#인풋 텍스트 -> 특정한 양식에 맞추어 정제/답변
def parsing():
    #CS 고객이 문의한 내용을 특정한 양식에 맞춰 뽑아내기/변형하기
    #해적 손님의 메일...
    customer = '''
        Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls \
        with smoothie! And to make matters worse, the warranty don't cover the cost of \
        cleaning up me kitchen. I need yer help right now, matey!
                '''
    #gpt야 해적손님의 메일을 격식을 갖춘 따뜻한 어투의 영어 메일로 바꿔줘.
    #gpt-4o 모델과 temperate 0.5로 맏는 채팅방을 연 상태(채팅방 = chat)
    chat = get_chat()

    #gpt에게 위의 손님 메일 + (변형) 요청을 보내, 답변을 받아오는 체인 정의
    style_chain = build_style_chain(chat)

    result = style_chain.invoke({'style' : 'Korean in a calm and respectful tone', 'text' : customer})
    print(result)

    text = input(f'{result}에 대한 나의 답변')
    reply = style_chain.invoke({'style' : 'english in a calm and respectful tone if my response contains some bad words, plz translate it or remove it', 'text' : customer})
    print(reply)



from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
def build_review_chain(chat):
    #1. prompt를 만드시오(build_style_chain 참고)
    # AutoTokenizer.from_pretrained(모델이름)
    prompt = ChatPromptTemplate.from_template(T.REVIEW_TEMPLATE)

    #**StrOutPutParser는 '거의 아무것도 하지 않는' str만 추출하는 역할
    #StructuredOutputParser 은 gpt가 생성한 str을 특정한 구조(자료형)로 파싱
    #schemas -> 나 이런 정보 필요해 ~ name(변수이름)
    schemas = [
        ResponseSchema(name="gift",
                       description="Was the item purchased as a gift for someone else? "
                                   "Answer True if yes, False if not or unknown."),
        ResponseSchema(name="delivery_days",
                       description="How many days did it take for the product to arrive? "
                                   "If this information is not found, output -1."),
        ResponseSchema(name="price_value",
                       description="Extract any sentences about the value or price, "
                                   "and output them as a comma separated Python list."),
    ]
    #리뷰를 합친 prompt가 인풋 -> chat이 이를 확인 => gpt(chat)가 생성한 결과를 schemas에 따라 구조화
    return prompt | chat | StructuredOutputParser.from_response_schemas(schemas), StructuredOutputParser.from_response_schemas(schemas).get_format_instructions()

#OutputParser의 종류를 바꿔서 Parser의 역할을 확인
#리뷰 -> 리뷰 여기저기에 존재하는 정보를 Parser가 골라내서 정리해주는 역할
def output_parsing():
    customer_review = """
    This leaf blower is pretty amazing. It has four settings: candle blower, gentle breeze, 
    windy city, and tornado. It arrived in two days, just in time for my wife's anniversary 
    present. I think my wife liked it so much she was speechless. It's slightly more expensive \
    than the other leaf blowers out there, but I think it's worth it for the extra features.
    """

    chat = get_chat()
    parse_chain, format = build_review_chain(chat)

    #build_review_chain이 시작할 때 필요한 재료가 들어가야함
    output = parse_chain.invoke({'text' : customer_review, 'format_instructions' : format})
    print(f'구조화된 파싱 : {type(output).__name__, output}')
    print(f'구조화 결과 delivery : {output.get('delivery_days')}')

def Legacy_chat():
    Chat = OpenAI()
    #Chat = ChatOpenAI(temperature = 0.6, model = MODEL_NAME)

    response = Chat.completions.create(
        model = MODEL_NAME,
        #role : system(openai의 세팅), user(사용자)
        messages = [{'role' : 'system', 'content' : '말끝에다가 멍을 붙여라'},
                    {'role' : 'user', 'content' : '한국은 어떤 ㅏ나라이니?'}],
        #답변의 창의성(0~1) 1에 가까울 수록 창의적(랜덤한) 답변
        temperature = 0.6
    )

    print(response)
    print(response.choice[0].message.content)


#LCEL으로 넘겨주기 위해 Runnable 패밀리 사용
#https://modulabs.co.kr/community/momos/284/feeds/3525
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
def build_seq_chain(llm):
    destination_chains = {
        #p가 뭘까를 생각해보면 좋음
        p['name'] : ChatPromptTemplate.from_template(p['prompt_template'])
        | llm
        | StrOutputParser() for p in T.PROMPT_INFOS
    }
    # print(destination_chains)
    #destination_chains는 4개의 페르소나를 가진 질의응답 체인이다.
    #default_chain : 일반응답을 수행해주는 일반 gpt
    default_chain = ChatPromptTemplate.from_template('{input}') | llm | StrOutputParser()

    destination_str = '\n'.join(f'{p['name']} : {p['description']}' for p in T.PROMPT_INFOS)
    router_prompt = ChatPromptTemplate.from_template(
        #MULTI_PROMPT_ROUTER_TEMPLATE 질문이 들어왔을 때 넷 중 하나의 체인을 고르라는 얘기
        #MULTI_PROMPT_ROUTER_TEMPLATE 이 가지고 있는 '채워줘야할 공백' => destinations
        #destination_str -> 네 개의 체인이 어떤 역할을 하는 체인인지 설명 PROMPT_INFOS
        T.MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations = destination_str)
    )

    #라우터 프롬프트 -> llm -> Json파서
    #JsonOutputParser가 '''코드펜스까지 벗겨줌
    router = router_prompt | llm | JsonOutputParser()

    verbose = True
    def route(info):
        #어느 체인이 답할지에 대한 정보를 가지고 (해당하는) 체인이 답변하게 하는 코드 ->route
        destination = info.get('dstination', 'DEFAULT')
        chain = destination_chains.get(destination, default_chain)
        if verbose:
            print(f'[router] : {destination} -> {info['next_inputs']}')
        return chain.invoke({'input' : info['next_inputs']})
    return router | RunnableLambda(route)

if __name__ == '__main__':
    # output_parsing()
    llm = get_chat()
    router = build_seq_chain(llm)
    questions = ['수학에서 원주가 뭐야?', '왜 건물들은 지진이 나도 안 무너져?', '가장 빠른 자료구조가 뭘까?']
    for q in questions:
        print(f'question : {q}\n answer : {router.invoke({'input' : q})[:200]}')