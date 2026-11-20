from agent.base import *

with open("prompt/personality_dimention_extract.prompt", 'r', encoding='utf-8') as f:
    personality_dimention_str = f.read()

class PersonalityDimensionOutput(BaseModel):
    personality_dimention_dic: dict[str, float] = Field(description="性格维度分数")

structured_llm = llm.with_structured_output(PersonalityDimensionOutput, method="function_calling")

import json
def personality_dimention_node(personality_preference_dic: dict) -> dict:
    input_str = json.dumps(personality_preference_dic,  ensure_ascii=False)

    messages = [
        {"role": "system", "content": personality_dimention_str},
        {"role": "user", "content": f"请处理以下 JSON 数据：\n{input_str}"},
    ]

    result = structured_llm.invoke(messages)
    return result.personality_dimention_dic