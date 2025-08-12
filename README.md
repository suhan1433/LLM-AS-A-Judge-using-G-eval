# LLM-AS-A-Judge-using-G-eval
LLM-AS-A-Judge using G-eval


## 사용법
1. requirements 설치
   ```
   pip install -r requirements.txt
   ```
2. FastAPI 접속
   ```
   uvicorn retrieve_api:app --reload
   ```
   
3. Swagger 접속
   ```
   주소/docs 
   ```
   
4. Data Format:
   ```Python
   [ {"query": str, "answer": str, "chunks": List[str]}, ...]
   ```
   
5. Swagger 파라미터 수정
   - Input_path  : LLM 평가에 사용할 입력 JSON 파일 경로
   - output_path : LLM 평가 결과를 저장할 JSON 파일 경로
   - llm_model   : 사용할 OpenAI LLM 모델명 (예: 'gpt-4o-mini')
   - limit_candidate_idx : 각 질문별로 평가할 후보 청크의 최대 개수 (상위 N개만 평가)
   - categorical_criteria : 범주형 평가지표의 인덱스이며, 리스트는 1부터 시작 (예: [1]), 범주형은 점수 합산 때 포함 안함, 만약 없는 경우 '[]' 입력
   - openai_api_key : OpenAI API 키 (입력하지 않으면 환경변수 OPENAI_API_KEY 사용)
   - system_prompt : SystemPrompt내용이며, Swagger보다 py에서 수정하는걸 권장
<img width="1442" height="512" alt="스크린샷 2025-08-11 오후 10 44 44" src="https://github.com/user-attachments/assets/4d0447a9-f0e2-4d9f-8fbf-f723d4050d29" />


6. 각 파라미터 설명 맨 아래 스키마에서 확인 가능
<img width="729" height="511" alt="스크린샷 2025-08-11 오후 10 44 57" src="https://github.com/user-attachments/assets/5a6d80df-f696-4fba-a271-f927f1fa14d0" />
