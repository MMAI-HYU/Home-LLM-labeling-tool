#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

SYSTEM_PROMPT = "당신은 AI 홈 어시스턴트입니다. 지시대로 작업을 수행하고, 주어진 정보로만 질문에 답하세요."
CTX_SIZE = 512

def tokenize(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=CTX_SIZE)

# -------------------------------------------------------------------
# 1. 한국어 프롬프트 및 시스템 메시지 정의
# -------------------------------------------------------------------
# 이전 대화에서 정의했던 한국어 버전 프롬프트들을 가져옵니다.
KOREAN_SYSTEM_PROMPT = "당신은 AI 홈 어시스턴트입니다. 지시대로 작업을 수행하고, 주어진 정보로만 질문에 답하세요."
KOREAN_TIME_PROMPT = "현재 시간과 날짜는"
KOREAN_SERVICES_PROMPT = "서비스:"
KOREAN_DEVICES_PROMPT = "기기:"
KOREAN_REQUEST_PROMPT = "사용자 요청:"
KOREAN_RESPONSE_PROMPT = "응답:"
KOREAN_BABEL_LOCALE = "ko_KR"
KOREAN_BABEL_FORMAT = "yyyy년 MMMM d일 EEEE a h시 m분"


# -------------------------------------------------------------------
# 2. 새로운 한국어 포맷 함수
# -------------------------------------------------------------------
def format_example_korean(example, device_map, current_dt):
    """
    주어진 데이터를 기반으로 한국어 프롬프트를 생성합니다.
    - example: 상태, 사용 가능 도구, 질문이 담긴 딕셔너리
    - device_map: 기기 ID와 한글 설명이 매핑된 딕셔너리
    - current_dt: 현재 시간을 나타내는 datetime 객체
    """
    # 시스템 프롬프트
    sys_prompt = KOREAN_SYSTEM_PROMPT

    # 현재 시간 정보 (Babel 라이브러리 사용)
    time_block = f"{KOREAN_TIME_PROMPT} {format_datetime(current_dt, KOREAN_BABEL_FORMAT, locale=KOREAN_BABEL_LOCALE)}"
    
    # 서비스 정보
    services_block = f"{KOREAN_SERVICES_PROMPT} " + ", ".join(sorted(example["available_tools"]))
    
    # 기기 상태 정보 (한글 설명 추가)
    states_list = []
    for state_string in example["states"]:
        device_id, state = state_string.split(' = ')
        # device_map에서 한글 설명을 찾아 추가합니다. 없으면 ID를 그대로 사용합니다.
        korean_name = device_map.get(device_id, device_id)
        formatted_state = f"{device_id} '{korean_name}' = {state}"
        states_list.append(formatted_state)
    states_block = KOREAN_DEVICES_PROMPT + "\n" + "\n".join(states_list)
    
    # 사용자 요청
    question = f"{KOREAN_REQUEST_PROMPT}\n" + example["question"]
    
    # 응답 시작
    response_start = KOREAN_RESPONSE_PROMPT

    # 최종 프롬프트 조합
    return "\n".join([sys_prompt, time_block, services_block, states_block, question, response_start])


def generate(model, tokenizer, prompt):
    eos_token_id = tokenizer(tokenizer.eos_token)["input_ids"][0]

    inputs = tokenize(tokenizer, prompt)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.15,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )
    text = tokenizer.batch_decode(outputs)
    return text

def format_example(example):
    sys_prompt = SYSTEM_PROMPT
    services_block = "서비스: " + ", ".join(sorted(example["available_tools"]))
    states_block = "기기:\n" + "\n".join(example["states"])
    question = "Request:\n" + example["question"]
    response_start = "Response:\n"

    return "\n".join([sys_prompt, services_block, states_block, question, response_start])


def main():
    model_folder = "models/korean-qwen-rev3-demo2/checkpoint-76"
    num_examples = 1
    actions = json.load(open('toyota_data/korean_captions_transformers.json', 'r'))

    torch.set_default_device("cuda")
    print(f"Loading model from {model_folder}...")
    trained_model = AutoModelForCausalLM.from_pretrained(model_folder, trust_remote_code=True, torch_dtype=torch.bfloat16)
    trained_tokenizer = AutoTokenizer.from_pretrained(model_folder, trust_remote_code=True)

    ret = []
    for vid in tqdm(list(actions.keys())):
        request = f"{actions[vid]}. 사용자의 편의성을 위한 적절한 조치를 취하세요."

        example = {
          "states": [
      "lightSwitch.room1_1 '거실 조명 스위치 1' = on",
	  "kitchenTv.main_1 '주방 TV' = on",
	  "smartSwitch.room1_1 '거실 스마트 스위치 1' = off",
	  "lighting.room1_2 '거실 조명 2' = off;seagreen (47, 111, 119);3%",
	  "smartGarageDoorOpener '차고 문 제어기' = closed",
	  "smartPowerPlug.room1_1 'Room1 1' = on",
	  "elevatorController '엘리베이터 제어기' = idle",
	  "smartLedBulb.room1_1 '거실 스마트 전구 1' = off;tan (216, 174, 142);97%"
          ],
          "available_tools": [
	  "elevatorController.call()", 
	  "kitchenTv.decrease_volume()", 
	  "kitchenTv.increase_volume()", 
	  "kitchenTv.set_channel()", 
	  "kitchenTv.set_volume()", 
	  "kitchenTv.turn_off()", 
	  "kitchenTv.turn_on()", 
	  "light.toggle()", 
	  "light.toggle()", 
	  "light.turn_off()", 
	  "light.turn_off()", 
	  "light.turn_on(brightness,rgb_color)", 
	  "light.turn_on(brightness,rgb_color)", 
	  "smartGarageDoorOpener.close()", 
	  "smartGarageDoorOpener.open()", 
	  "smartGarageDoorOpener.stop()", 
	  "smartGarageDoorOpener.toggle()", 
	  "smartPowerPlug.toggle()", 
	  "smartPowerPlug.turn_off()", 
	  "smartPowerPlug.turn_on()", 
	  "smartSwitch.toggle()", 
	  "smartSwitch.toggle()", 
	  "smartSwitch.turn_off()", 
	  "smartSwitch.turn_off()", 
	  "smartSwitch.turn_on()", 
	  "smartSwitch.turn_on()"
          ],
          "question": request
        }

        prompt = format_example(example)

        print(prompt)

        output = generate(trained_model, trained_tokenizer, [ prompt for x in range(num_examples) ])

        outs = []

        for text in output:
            outs.append(text.replace(trained_tokenizer.eos_token, ""))
            print(outs[-1])

        ret.append({
            'video_id':vid,
            'action':actions[vid],
            'response':outs
            })

    json.dump(ret, open('0917-response.json', 'w'))


if __name__ == "__main__":
    main()
