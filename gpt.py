import json
import jsonlines
from openai import OpenAI
from tqdm import tqdm

# GPT API 클라이언트 생성
client = OpenAI(api_key="API_KEY")

def get_sys_prompt(ACTION):

    sys_prompt="""# 역할 및 목적
    - 본 AI는 LLM 학습을 위한 데이터 라벨링 작업을 지원합니다. 사용자는 assistant 역할로 적합한 한글 답변을 작성해야 합니다.

    # 지침
    - 입력된 system prompt의 내용을 명확하게 파악합니다.
    - 사용자의 행동 목적과 편의 요청을 분석합니다. 이때, "request"의 "value"는 "사용자가 <행동> 중입니다. 사용자의 편의성을 위한 적절한 조치를 취하세요."로 작성합니다.\n"""+\
    f'- <행동>은 {ACTION}를 사용합니다.\n'+\
    """- system prompt의 디바이스 리스트를 참고하여 적합한 디바이스 제어 및 답변을 준비합니다.
    - 답변은 예시와 동일한 구조(assistant 응답 포맷)로 작성합니다.

    # 작업 전 계획
    - 다음 항목으로 이 작업을 처리하세요:
      1. 입력 프롬프트의 목적과 요청 파악
      2. 디바이스 리스트 분석
      3. 사용자 행동에 맞는 조치 정하기
      4. assistant 답변 포맷에 맞게 응답 작성
      5. 중복 출력을 방지하며, 명시된 디바이스 이름만 사용

    # 세부 규칙
    - "request"는 사용자의 행동을 한 문장으로 작성하고, 이어서 사용자의 편의를 위한 조치를 요청하는 형태입니다.
    - "assistant"는 system prompt의 디바이스 리스트를 참고하여 올바른 디바이스 제어를 수행하고, 응답을 출력해야 합니다.
    - 출력은 반드시 아래 예시처럼 JSON 객체 배열 형태로 구성합니다. 중복된 샘플 출력을 피하세요.

    # 예시
    [
      {
        "from": "request",
        "value": "사용자가 주방에서 요리 중입니다. 사용자의 편의성을 위해 적절한 조치를 취하세요."
      },
      {
        "from": "assistant",
        "value": "주방에서 요리 중이시므로 TV를 켜드릴게요.\n```homeassistant\n{\"service\": \"kitchenTv.turn_on\", \"target_device\": \"kitchenTv.main_1\"}\n```"
      }
    ]

    # 추가 설명
    - 디바이스 리스트가 명시된 경우, assistant 응답에 반드시 반영하세요. 예: {"service": "device.service_name", "target_device": "device.device_id"}
    - 어떤 상황에서든 추가 문장을 임의로 출력하지 않습니다."""

    return sys_prompt

# JSON 파일 로드 함수
def load_json(file_path):
    with jsonlines.open(file_path) as f:
        data = [line for line in f.iter()]
    return data

# GPT 호출 함수
def call_gpt(prompt):
    action = ['마시기', '앉기', '책 읽기', '일어나기', 'TV 시청하기', '먹기', '요리하기', '스마트폰 사용하기', '노트북 사용하기', '청소하기']
    import random
    random.shuffle(action)
    print(action[0])
    sys_prompt = get_sys_prompt(action[0])
    response = client.chat.completions.create(
        model="gpt-5-mini",  # 원하는 모델로 변경
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

def main():
    # JSON 파일 읽기
    count=1000
    start=200
    data = load_json("data/korean/home_assistant_train.jsonl")[start:start+count]
    output_jsonl = 'data/korean/gpt_gerenated_train.jsonl'

    # list 안의 값들을 순회하면서 GPT API 호출
    for i, item in tqdm(enumerate(data), total=len(data)):
        print(f"=== Item {i+1} ===")
        prompt = str(item['conversations'][0])  # 혹은 가공 가능
        answer = call_gpt(prompt)
        try:
            json_answer = json.loads(answer)
        except:
            continue
        json_answer[0]['from'] = 'user'
        ret = {'conversations' : [item['conversations'][0],
                json_answer[0],
                json_answer[1]]}

        with open(output_jsonl, 'a', encoding="utf-8") as f:
            json.dump(ret, f, ensure_ascii=False)
            f.write('\n')

        print("Prompt:", prompt)
        print("Answer:", answer)
        print()

if __name__ == "__main__":
    main()
