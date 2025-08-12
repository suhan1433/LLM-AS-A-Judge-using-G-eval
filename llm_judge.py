from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import requests
import json
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import re
import numpy as np
from collections import namedtuple
from typing import List, Dict, Any, Tuple
import codecs

app = FastAPI()

# ----------- 3. LLM Judge Endpoint -----------
class LLMJudgeRequest(BaseModel):
    input_path: str = Field(
        'query_answer_chunks_data.json',
        description="LLM 평가에 사용할 입력 JSON 파일 경로"
    )
    output_path: str = Field(
        'llm_judge_result.json',
        description="LLM 평가 결과를 저장할 JSON 파일 경로"
    )
    llm_model: str = Field(
        'gpt-4o-mini',
        description="사용할 OpenAI LLM 모델명 (예: 'gpt-4o-mini')"
    )
    limit_candidate_idx: int = Field(
        1,
        description="각 질문별로 평가할 후보 청크의 최대 개수 (상위 N개만 평가)"
    )
    categorical_criteria: Optional[list[int]] = Field(
        [1],
        description="범주형 평가지표의 인덱스 리스트 1부터 시작 (예: [1]), 범주형은 점수 합산 때 포함 안함, 만약 없는 경우 '[]' 입력" 
    )
    openai_api_key: Optional[str] = Field(
        None,
        description="OpenAI API 키 (입력하지 않으면 환경변수 OPENAI_API_KEY 사용)"
    )
    system_prompt: str = Field(
        """
당신은 주어진 골든 답안(Golden Answer)과 질문(Question)을 활용하여 문서 청크(Chunk)를 평가하여 골든 청크(Golden Chunk)를 정하는 전문 평가자입니다.
골든 답안(Golden Answer)은 여러 문서청크(Chunk)로 분산되어 있을 수 있으며, 각 청크가 골든 답안의 일부분을 담고 있는지 판단해야 합니다.

**주요 임무:** 아래의 EVALUATION CRITERIA에 따라 각 문서 청크(Chunk), 골든 답안(Golden Answer), 질문(Question)을 활용하여 평가하세요.
---

## EVALUATION CRITERIA:

### 1. **GOLDEN CHUNK IDENTIFICATION (골든 청크 여부) (Score 1 or 2)**
   - 청크가 골든 답안(Golden Answer)의 내용을 '일부라도' 포함하고 있으면 골든청크로 간주합니다.
   - 반드시 골든 답안의 전체 내용을 모두 포함할 필요는 없습니다.
   - 평가 항목:
     * 문서 청크가 골든 답안의 핵심 정보 중 일부라도 포함하고 있는가?
     * 문서 청크가 골든 답안의 내용을 부분적으로라도 담고 있다면 골든청크로 간주합니다.
   - 평가 절차:
     1. 청크의 정보가 골든 답안의 정보를 직접적으로 포함하고 있는지 확인합니다.
     2. 아래 기준에 따라 1점(골든 청크 간주할 수 없음) 또는 2점(골든 청크 간주 가능) 중 하나를 부여합니다.

   **점수 기준:**
   - **2점:** 청크가 골든 답안의 내용을 일부라도 포함하고 있으면 골든청크로 간주 (즉, 골든 답안의 핵심 정보 중 일부만 포함해도 2점)
   - **1점:** 청크가 골든 답안의 내용을 전혀 포함하고 있지 않으면 골든청크로 간주하지 않음

---

### 2. **GOLDEN CONTENT COVERAGE (골든 답안 내용 포함도) (Score 1-5)**
   - 청크가 골든 답안의 핵심 정보를 얼마나 포함하고 있는지 평가
   - 평가 항목:
     * 골든 답안의 특정 단계/부분을 담고 있는가
     * 골든 답안의 핵심 키워드나 개념이 포함되어 있는가
     * 부분적이라도 골든 답안 구성에 필수적인 정보인가
   - 평가 절차:
     1. 문서 청크와 골든 답안을 비교합니다.
     2. 청크 내에 골든 답안의 핵심 정보(키워드, 단계, 개념 등)가 얼마나 포함되어 있는지 확인합니다.
     3. 포함된 정보의 양과 질을 기준으로 점수 기준표에 따라 1-5점 중 하나를 부여합니다.

   **점수 기준:**
   - **5점:** 골든 답안의 핵심 부분을 완전히 포함하며, 해당 단계/섹션을 완벽히 설명함
   - **4점:** 골든 답안의 중요한 부분을 대부분 포함하지만, 일부 세부사항 누락
   - **3점:** 골든 답안의 일부분을 포함하지만, 불완전하거나 부분적임
   - **2점:** 골든 답안과 약간의 연관성은 있으나, 핵심 내용이 부족함
   - **1점:** 골든 답안과 거의 관련 없거나 중요도가 매우 낮음

---

### 3. **INFORMATION COMPLETENESS (정보 완성도) (Score 1-5)**
   - 청크 내 정보의 완성도와 자립성 평가
   - 평가 항목:
     * 청크 단독으로도 의미가 완전한가
     * 필요한 맥락 정보가 충분히 포함되어 있는가
     * 단계별 설명의 경우 해당 단계가 완전히 설명되어 있는가
   - 평가 절차:
     1. 청크만 단독으로 읽고, 맥락이나 추가 정보 없이도 이해가 가능한지 확인합니다.
     2. 필요한 배경 정보, 맥락, 단계별 설명이 충분히 포함되어 있는지 점검합니다.
     3. 정보의 완성도와 자립성을 기준으로 점수 기준표에 따라 1-5점 중 하나를 부여합니다.

   **점수 기준:**
   - **5점:** 청크 단독으로도 완전한 정보를 제공하며, 맥락이 명확함
   - **4점:** 대부분 완전하지만, 일부 맥락 정보가 부족함
   - **3점:** 기본적인 정보는 포함하지만, 완성도가 다소 부족함
   - **2점:** 정보가 단편적이며, 다른 정보와 연결되어야 이해 가능함
   - **1점:** 정보가 매우 불완전하며, 단독으로는 의미 파악이 어려움

---

### 4. **FACTUAL ACCURACY (사실적 정확성) (Score 1-5)**
   - 골든 답안과 청크를 비교했을 때의 사실적 정확성 평가
   - 평가 항목:
     * 골든 답안과 모순되는 정보가 있는가
     * 수치, 절차, 방법 등이 정확한가
     * 잘못된 정보나 왜곡된 내용이 포함되어 있는가
   - 평가 절차:
     1. 청크의 모든 정보(사실, 수치, 절차 등)를 골든 답안과 대조합니다.
     2. 모순, 오류, 왜곡된 내용이 있는지 확인합니다.
     3. 사실적 정확성의 기준으로 점수 기준표에 따라 1-5점 중 하나를 부여합니다.

   **점수 기준:**
   - **5점:** 골든 답안과 완벽히 일치하며, 모든 사실이 정확함
   - **4점:** 대부분 정확하지만, 사소한 차이나 표현의 차이 존재
   - **3점:** 주요 사실은 맞지만, 일부 세부사항에서 차이 있음
   - **2점:** 중요한 사실에서 일부 오류가 있음
   - **1점:** 골든 답안과 상당한 차이가 있거나 잘못된 정보 포함

---

### 5. **RELEVANCE (연관성) (Score 1-5)**  
   - 질문과 청크의 관련성과 정합성을 평가합니다.
   - 평가 항목:  
     * 청크가 질문의 핵심 의도에 적절히 대응하는가 
     * 불필요한 정보가 포함되어 있지 않은가 
     * 답변이 질문의 초점을 잘 유지하고 있는가 
      
   **점수 기준:**  
   - **5점:** 질문과 매우 밀접하게 연관되며, 불필요한 정보가 전혀 없음.  
   - **4점:** 대체로 연관되지만, 일부 불필요한 정보가 포함됨.  
   - **3점:** 연관성이 있지만, 핵심 초점이 일부 흐려짐.  
   - **2점:** 질문과 다소 동떨어진 내용이 포함되며, 초점이 맞지 않음.  
   - **1점:** 질문과 대부분 관련이 없으며, 핵심적인 내용이 부족함.  

---


### **FORMAT YOUR EVALUATION AS FOLLOWS FOR EACH CRITERION:**  
각 평가마다 아래 포맷으로 평가를 작성해주세요.  
**아래 예시의 형식을 반드시 그대로 따르세요.**
```
=== CHUNK EVALUATION ===
GOLDEN CHUNK IDENTIFICATION
점수: 2
근거: (여기에 상세 근거)

=== CHUNK EVALUATION ===
평가 항목 제목 
점수: 5
근거: (여기에 상세 근거)
```
---
        """,
        description="LLM 평가에 사용할 시스템 프롬프트 (평가 기준 등 포함, 필수)"
    )

@app.post('/llm-judge')
def llm_judge(req: LLMJudgeRequest):
    load_dotenv()
    api_key =  os.getenv("OPENAI_API_KEY", req.openai_api_key)
    if not api_key:
        return {"status": "error", "message": "OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다."}
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()
    user_prompt = """
    질문(Question): {Question}\n정답(Answer): {Answer}\n문서 청크(Chunk): {Chunk}
    """
    with open(req.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    TopLogprob = namedtuple("TopLogprob", ["token", "bytes", "logprob"])
    ChatCompletionTokenLogprob = namedtuple("ChatCompletionTokenLogprob", ["token", "bytes", "logprob", "top_logprobs"])
    
    
    def _top_logprobs_to_probs(top_logprobs: List[TopLogprob]) -> Dict[str, float]:
        # 숫자(1~5) 토큰만 필터링
        filtered = [tlp for tlp in top_logprobs if len(tlp.bytes) == 1 and 49 <= tlp.bytes[0] <= 53]
        if not filtered:
            return {}
        logprobs = np.array([tlp.logprob for tlp in filtered], dtype=np.float64)
        probs = np.exp(logprobs - np.max(logprobs))
        probs = probs / np.sum(probs)
        return {tlp.token: round(prob, 2) for tlp, prob in zip(filtered, probs)}

    def _find_all_score_toplogprobs(logprobs_content: List[ChatCompletionTokenLogprob]) -> List[Dict[str, float]]:
        target_tokens = ['점', '수', ':', ' ']
        n = len(target_tokens)
        results = []
        i = 0
        while i <= len(logprobs_content) - n:
            # 시퀀스가 일치하는지 확인
            if [logprobs_content[i+j].token for j in range(n)] == target_tokens:
                score_token_logprob = logprobs_content[i + n]
                # print(f"점수 토큰: {score_token_logprob.token}")
                # print("Top logprobs (숫자 1~5만):")
                # print(score_token_logprob.top_logprobs)
                probs_dict = _top_logprobs_to_probs(score_token_logprob.top_logprobs)
                # for token, prob in probs_dict.items():
                #     print(f"  후보: {token}, 확률: {round(prob, 1)}")
                results.append(probs_dict)
                i += n + 1  # 다음 "점수:" 이후로 이동 (중복 방지)
            else:
                i += 1
        if not results:
            print("해당 시퀀스를 찾을 수 없습니다.")
        return results


    def _eval_score(content: str, result: List[Dict[str, float]]) -> List[float]:
        criteria = _extract_evaluations_from_content(content=content)
        result_score = []
        for i in range(len(criteria)):
            score = 0
            for point in range(1, 6):
                count = result[i].get(str(point), 0)
                score += round(count * point, 2)
            result_score.append(round(score, 2))
        return result_score


    def _tokens_to_content(token_logprobs: List[ChatCompletionTokenLogprob]) -> str:
        
        reason_content = ''.join(tlp.token for tlp in token_logprobs)
        bytes_token = codecs.decode(reason_content, 'unicode_escape')
        
        return bytes_token.encode('latin1').decode('utf-8')


    def _extract_evaluations_from_content(content: str) -> List[Tuple[str, str, str]]:
        # 평가 항목별로 분리
        pattern = r"=== CHUNK EVALUATION ===\s*([^\n]+)\s*점수:\s*([0-9]+)\s*근거:\s*([^\n]+(?:\n(?!=== CHUNK EVALUATION ===).*)*)"
        matches = re.findall(pattern, content, re.MULTILINE)
        return matches
        

    def _generate_evaluations(content: str, total_score: List[float], categorical_criteria: List[int]) -> List[Dict[str, Any]]:
        matches = _extract_evaluations_from_content(content=content)

        evaluations = []
        for i, (category, generated_score, reason) in enumerate(matches):
            reason = reason.strip()
            score =  int(generated_score) if i+1 in categorical_criteria else total_score[i] #범주형은 점수 제거 
            evaluations.append({
                "category": category.strip(),
                "score": score,
                "reason": reason
            })
        return evaluations

    def _load_json(path : str, default=None) -> Any:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default if default is not None else []

    def _save_json(path : str, data : Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    golden_results = _load_json(req.output_path, default=[])
    processed_queries = set(item["query"] for item in golden_results)

    for item in tqdm(data):
        question = item['query']
        if question in processed_queries:
            continue
        answer = item['answer']
        sum_scores = []
        evaluation_list = []
        score_prob_result_list = []
        chunks = item['chunks']
        if chunks:
            for idx, chunk in enumerate(chunks):
                if idx == req.limit_candidate_idx:
                    break
                prompt = user_prompt.format(
                    Question=question,
                    Answer=answer,
                    Chunk=chunk
                )
                response = client.chat.completions.create(
                    model=req.llm_model,
                    messages=[
                        {"role": "system", "content": req.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    logprobs=True,
                    top_logprobs=6, # 점수(1-5)토큰 확률 얻기 위해 5개 이상으로 설정
                    temperature=0.0, # 모델 결정성을 높이기 위해 0.0으로 설정
                )
                logprobs = response.choices[0].logprobs.content
                score_prob_result = _find_all_score_toplogprobs(logprobs_content=logprobs)
                score_prob_result_list.append(score_prob_result)
                content = _tokens_to_content(token_logprobs=logprobs)
                total_score = _eval_score(content=content, result=score_prob_result)
                sum_scores.append(round(sum(total_score[i] for i in range(len(total_score)) if i+1 not in req.categorical_criteria ), 2))
                evaluations = _generate_evaluations(content=content, total_score=total_score, categorical_criteria=req.categorical_criteria)
                evaluation_list.append(evaluations)
        try:
            golden_results.append({
                "query": question,
                "answer": answer,
                "chunks": item['chunks'],
                "score_prob_result": score_prob_result_list,
                "evaluations": evaluation_list,
                "total_score": sum_scores,
            })
            _save_json(req.output_path, golden_results)
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "success", "output_path": req.output_path, "count": len(golden_results)}
