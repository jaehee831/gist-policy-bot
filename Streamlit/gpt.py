import os
import numpy as np
import faiss
import openai
import time
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langdetect import detect

# .env 파일 로드
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    raise FileNotFoundError(".env 파일을 찾을 수 없습니다.")

# 환경 변수에서 API 키 가져오기
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
openai.api_key = api_key

# 현재 디렉토리 설정
current_dir = os.getcwd()
vector_db_dir = os.path.join(current_dir, '../VectorDB')

# FAISS 인덱스 로드
index = faiss.read_index(os.path.join(vector_db_dir, 'vector_db.index'))
index_dimension = index.d  # 인덱스의 차원 확인

# 파일 경로 로드
with open(os.path.join(vector_db_dir, 'file_paths.txt'), 'r', encoding='utf-8') as f:
    file_paths = [line.strip() for line in f]

# 모든 문서 로드
documents = []
for path in file_paths:
    with open(path, 'r', encoding='utf-8') as file:
        documents.append(file.read())

# 임베딩 생성 함수
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def search(query, top_k=3):
    # 쿼리 벡터화
    query_embedding = get_embedding(query).reshape(1, -1)
    
    # 차원 일치 여부 확인
    if query_embedding.shape[1] != index_dimension:
        raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {index_dimension}")

    # FAISS 인덱스에서 유사한 문서 검색
    D, I = index.search(query_embedding, top_k)
    
    results = [(file_paths[i], documents[i]) for i in I[0]]
    return results

def generate_answer(query, top_k=2):
    # 관련 문서 검색
    relevant_docs = search(query, top_k)
    
    # 관련 문서들을 하나의 문자열로 결합
    context = "\n\n".join(doc for _, doc in relevant_docs)
    references = "\n".join(path for path, _ in relevant_docs)
    
    language = detect(query)

    if language == 'ko':
        system_message = ("이 챗봇은 광주과학기술원(GIST) 구성원들의 규정 관련 질문에 대한 답변을 제공하기 위해 "
                          "만들어진 챗봇입니다. 챗봇은 데이터베이스에 있는 정보를 기반으로 질문에 답변합니다. "
                          "데이터베이스에 해당 정보가 없는 경우 다음과 같이 응답해야 합니다: [죄송합니다, 이 정보는 GIST 규정 데이터베이스에서 "
                          "찾을 수 없습니다. 해당 부서에 문의하여 도움을 받으시기 바랍니다. (GIST 기획팀 황인호 팀장 062-715-2971)]라는 "
                          "안내가 표시됩니다. 또한 사용자의 질문이 불완전한 경우 챗봇은 데이터베이스를 기반으로 추가 정보를 요청해야 합니다. "
                          "예를 들어 '보다 정확한 답변을 드리기 위해 몇 가지 추가 정보가 필요합니다. 필요한 추가 정보]와 같은 세부 정보를 "
                          "제공해 주시겠습니까?")
    else:
        system_message = ("This GPT is a chatbot designed to provide answers to questions related to regulations for members of the Gwangju Institute of Science and Technology (GIST). "
                          "It should respond in a friendly manner. The chatbot will answer questions based on information in the database. If the information is not available in the database, "
                          "it should respond as follows: [Sorry, this information is not available in the GIST regulations database. Please contact the relevant department for assistance. "
                          "(GIST 기획팀 황인호 팀장 062-715-2971)] Additionally, if the user's question is incomplete, the chatbot should request additional information based on the database. "
                          "For example: [To provide you with a more accurate answer, I need some additional information. Could you please provide details such as [necessary additional information]?")

    # OpenAI GPT-4 모델을 사용하여 답변 생성
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Here are some documents:\n\n{context}\n\nQuestion: {query}\nAnswer:"}
        ],
        max_tokens=1000  # max_tokens 값을 증가시켜 잘리는 문제 방지
    )
    
    answer = response.choices[0].message['content'].strip()
    return answer, references

# Streamlit 앱 설정
st.image(r"..\gist.jpg", use_column_width=True)
st.header("🤖 Gist Policy App (Demo)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('질문: ', '', key='input')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    answer, references = generate_answer(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append((answer, references))

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        user_question = st.session_state['past'][i]
        bot_answer, references = st.session_state['generated'][i]
        st.write(f'**질문:** {user_question}')
        st.write(f'**답변:** {bot_answer}')
        st.write(f'**참조한 문서 경로:**\n{references}')
