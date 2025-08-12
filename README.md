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
   ```JSON
   [ {"query": str, "answer": str, "chunks": List[str]}, ...]
   ```
   
5. Swagger 파라미터 수정
   - Input_path  :
   - output_path :
   - llm_model   :
   - limit_candidate_idx :
   - categorical_criteria
   - openai_api_key :
   - system_prompt :
<img width="1442" height="512" alt="스크린샷 2025-08-11 오후 10 44 44" src="https://github.com/user-attachments/assets/4d0447a9-f0e2-4d9f-8fbf-f723d4050d29" />


6. 각 파라미터 설명 맨 아래 스키마에서 확인 가능
<img width="729" height="511" alt="스크린샷 2025-08-11 오후 10 44 57" src="https://github.com/user-attachments/assets/5a6d80df-f696-4fba-a271-f927f1fa14d0" />
