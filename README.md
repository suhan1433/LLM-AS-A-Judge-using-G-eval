# LLM-as-a-judge-using-G-eval
LLM-as-a-judge 방법론을 OpenAI API를 활용하여 구현하되, 단순한 점수 출력 대신 각 점수에 대한 확률 분포를 추출하여 모델의 불확실성을 반영한 보다 정확한 평가 점수를 산출하는 방법

## Overview

이 방법론은 검색 성능 파이프라인 구축 과정에서 Question과 Answer만 주어진 상황에서 골든 청크(Golden Chunk)를 선정하기 위해 개발됨
기존 LLM-as-a-judge 방식의 한계를 보완하여 모델의 판단 불확실성을 정량적으로 반영한 평가 시스템을 구현함


## Problems of the existing LLM-as-a-Judge approaches

- 불확실성 미반영 문제

   - 단일 점수 출력의 한계: 모델이 최종적으로 생성한 점수(예: "4")만으로 평가할 경우, 해당 판단에 대한 모델의 확신도를 알 수 없음
   - 확률 분포 정보 손실: 모델이 다른 점수 후보들에 대해 어느 정도 가능성을 부여했는지에 대한 정보가 소실됨

- 구체적 문제 사례
   예시: 모델이 "4점"을 출력했을 때
   - 실제 확률 분포: "4"(0.62), "3"(0.38), "5"(0.00), "2"(0.00), "1"(0.00)
   - 기존 방식: "4점"만 반환 → 불확실성 정보 손실
   - 개선 방식: 가중 평균을 통한 정밀한 점수 산출 (4×0.62 + 3×0.38 = 3.62점)
     <img width="538" height="392" alt="image" src="https://github.com/user-attachments/assets/17418780-5206-42de-8923-cb6fc08a9044" />
     
- 기존 평가 도구의 제약
DeepEval 등 기존 평가 프레임워크는 다음과 같은 한계가 있음:

   - 모델이 의도한 토큰을 실제로 생성했는지 검증 불가
   - 각 점수 토큰에 대한 확률 분포 정보 접근 불가
   - 평가 과정의 투명성 및 제어 가능성 부족

### 솔루션: 확률 분포 기반 평가
OpenAI API의 logprobs 기능을 직접 활용하여:

1. 각 점수 토큰의 확률 분포 추출
2. 확률 가중 평균을 통한 최종 점수 계산
3. 모델 불확실성을 정량적으로 반영한 보다 신뢰할 수 있는 평가 결과 제공

이를 통해 단순한 점수 출력을 넘어서 모델의 판단 과정과 확신도까지 고려한 LLM-as-a-judge 평가 시스템을 구현




## Quick Start
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
   - ```Input_path```  : LLM 평가에 사용할 입력 JSON 파일 경로
   - ```output_path``` : LLM 평가 결과를 저장할 JSON 파일 경로
   - ```llm_model```   : 사용할 OpenAI LLM 모델명 (예: 'gpt-4o-mini')
   - ```limit_candidate_idx``` : 각 질문별로 평가할 후보 청크의 최대 개수 (상위 N개만 평가)
   - ```categorical_criteria``` : 범주형 평가지표의 인덱스이며, 리스트는 1부터 시작 (예: [1]), 범주형은 점수 합산 때 포함 안함, 만약 없는 경우 '[]' 입력
   - ```openai_api_key``` : OpenAI API 키 (입력하지 않으면 환경변수 OPENAI_API_KEY 사용)
   - ```system_prompt``` : SystemPrompt내용이며, Swagger보다 py에서 수정하는걸 권장(Template는 수정하면 안됩니다)
<img width="1442" height="512" alt="스크린샷 2025-08-11 오후 10 44 44" src="https://github.com/user-attachments/assets/4d0447a9-f0e2-4d9f-8fbf-f723d4050d29" />


6. 각 파라미터 설명 맨 아래 스키마에서 확인 가능
<img width="729" height="511" alt="스크린샷 2025-08-11 오후 10 44 57" src="https://github.com/user-attachments/assets/5a6d80df-f696-4fba-a271-f927f1fa14d0" />

## Input/Output

### Input

- query_answer_chunks_data.json

```
{
    "query": "웹사이트 관리자 콘솔에 로그인하려면 어떻게 해야 하나요?",
    "answer": "웹사이트 관리자 포털 로그인 페이지에서 [사용자명], [비밀번호]를 입력 후 [로그인] 버튼을 클릭합니다.  \n로그인 성공 시 '관리자 대시보드' 화면으로 이동합니다.  \n경로: [관리자 대시보드] > [로그인]",
    "chunks": [
      "웹사이트 관리자 콘솔을 사용하기 위해서는 관리자 포털 화면에서 로그인을 해야 합니다.\n- 관리자 포털 로그인 페이지 접속 후 [사용자명, 비밀번호] 입력 후 [로그인] 버튼 클릭\n\n  관리자 콘솔의 로그인 화면. 사용자명과 비밀번호 입력 필드, 그리고 로그인 버튼이 표시되어 있습니다.\n\n- 로그인 성공 시 관리자 대시보드 화면으로 이동\n\n  관리자 대시보드 화면. 사이트 통계 그래프, 일일 사용자 트래픽, 콘텐츠 현황 목록, 시스템 알림, 관리 작업 목록이 표시되어 있습니다.",
      "콘텐츠 유형을 먼저 선택해야 합니다. 저장소 체크박스: [하단 좌측] - 배포에 사용할 콘텐츠 저장소를 선택합니다. 저장소 라벨: [하단 중앙] - 선택된 콘텐츠 저장소의 이름을 표시합니다. 표시된 데이터/텍스트는 다음과 같습니다. 배포 워크플로우:[표준] 배포 워크플로우 선택 드롭다운의 기본 값으로 선택되어 있습니다.워크플로우 목록으로는 \"콘텐츠검토-승인-스테이징배포-품질검사\", \"표준_배포\", \"검토, 예약배포\", 그리고 \"검토 요청 예약\"이 배포 워크플로우 선택 드롭다운 메뉴에 표시된 워크플로우 이름입니다. * 즉시 : 기존 실행하고 있던 표준 워크플로우를 강제 중단 후 즉시 사용되는 워크플로우 배포 워크플로우 설정 화면. 콘텐츠 배포 작업을 위한 워크플로우를 정의하고 설정하는 화면입니다. 배포 워크플로우 유형을 선택하고, 담당자를 지정하며, 콘텐츠 저장소 및 버전을 선택하여 배포 프로세스를 구성합니다. 주요 화면 요소는 다음과 같습니다."
    ]
  },
```

### Output
| **필드명**             | **타입**      | **설명**                    | **예시 (값)**                                                                                                 |
| ------------------- | ----------- | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| category            | str         | 각 평가 기준 제목                  | GOLDEN CONTENT COVERAGE                                                                                    |
| score               | float       | 각 평가 결과 점수(0-5점 사이)         | 4.47 (4x0.53 + 5x0.47)                                                                                                       |
| reason              | str         | 각 평가 결과                     | 청크는 골든 답안의 주요 단계인 로그인 과정과 성공 후 이동하는 화면에 대한 정보를 대부분 포함하고 있지만, '로그인 페이지'에 대한 구체적인 설명이 부족하여 일부 세부사항이 누락되었습니다. |
| score\_prob\_result | List\[dict] | 각 평가 점수 확률 분포             | {<br>  "4": 0.53,<br>  "5": 0.47,<br>  "3": 0.0,<br>}                        |
| total\_score        | float       | 모든 평가 점수 확률분포를 고려해 합산한 점수 | 6.47                                                                                                       |
- llm_judge_result.json

```python
    "score_prob_result": [
      [ # 첫번째 청크 LLM-JUDGE 결과
        { # GOLDEN CHUNK IDENTIFICATION 평가지 확률분포
          "2": 1.0,
          "1": 0.0
        },
        { # "GOLDEN CONTENT COVERAGE" 평가지 확률 분포
          "4": 0.53,
          "5": 0.47,
          "3": 0.0
        },
     ],
     [ # 두번째 청크 LLM-JUDGE 결과
      ...
     ]
   ],
   "evaluations": [
      [ # 첫번째 청크 LLM-JUDGE 결과
        {
          "category": "GOLDEN CHUNK IDENTIFICATION",
          "score": 2,
          "reason": "문서 청크는 골든 답안...
        },
        {
          "category": "GOLDEN CONTENT COVERAGE",
          "score": 4.47,
          "reason": "청크는 골든 답안의 주요...
        },
      ],
      [ # 두번째 청크 LLM-JUDGE 결과
         ...
      ]
   ],
  "total_score": [
      6.47, # 첫번째 청크 LLM-JUDGE 총 점수 (2 + 4.47) -> 만약 "GOLDEN CHUNK IDENTIFICATION" 범주형으로 설정한다면 점수 합산에서 제외
      2.1   # 두번째 청크 LLM-JUDGE 총 점수
    ]
```

## LLM-judge G-Eval Method
**1. Openai-api의 logprob 기능을 통해 각 평가의 ```점수 토큰```일 때의 logprob를 선택**

예를 들어, "점수 : 4"와 같이 점수를 생성할 때, "점수 :"까지의 토큰을 먼저 생성한 후,
그 다음에 점수(예: '4') 토큰이 생성되는 시점의 top logprob 결과를 확인 가능

출력 예시:
```
TopLogprob(token='4', bytes=[52], logprob=-0.47439804673194885), 
TopLogprob(token='3', bytes=[51], logprob=-0.9743980169296265), 
TopLogprob(token='5', bytes=[53], logprob=-8.099397659301758), 
TopLogprob(token='2', bytes=[50], logprob=-10.974397659301758), 
```

**2. 각 점수에 해당하는 토큰의 logprob에 대해 exp 연산 연산 후, 이 값들의 합으로 나누어 정규화 진행**
   
$$P(\text{token}) = \frac{\exp(\log \text{prob}(\text{token}))}{\sum_i \exp(\log \text{prob}(\text{token}_i))}$$

**3. 최종 점수**
   
각 점수 토큰의 확률에 해당 점수를 곱해 모두 더한 기대값(가중 평균)으로 산출

점수=1×P(1)+2×P(2)+3×P(3)+4×P(4)+5×P(5)


```
# 소수점 2째 자리까지 반올림
토큰 '4': 약 0.6223 -> 0.62
토큰 '3': 약 0.3774 -> 0.38
토큰 '5': 약 0.0003 -> 0
토큰 '2': 약 0.000017 -> 0
```
점수 = (4x0.62) + (3x0.38)

<img width="538" height="392" alt="image" src="https://github.com/user-attachments/assets/17418780-5206-42de-8923-cb6fc08a9044" />



**5. 순위 및 골든청크 선정 방법**

각 평가점수들의 합산 또는 규칙을 주어 순위 또는 골든청크 선정 할 수 있음
```
GOLDEN CHUNK IDENTIFICATION + GOLDEN CONTENT COVERAGE, # 각 점수의 합

GOLDEN CHUNK IDENTIFICATION >= 1.5 and GOLDEN CONTENT COVERAGE >= 3.5 # 규칙 기반
```



## 소요 시간 및 비용
| 항목 | 내용 |
| --- | --- |
| 모델 | GPT-4o-mini |
| 데이터 |  [Question, Answer, Chunk] 290쌍 |
| 입력 크기 | 시스템 프롬프트 + 질문(40자 내외) + 답변(350자 내외) + 청크(500자 내외) |
| 출력 크기 | 약 1,100자 |
| 총 API 비용 | 0.15 달러 |
| 총 소요 시간 | 37분 |
