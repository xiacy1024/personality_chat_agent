from agent.base import *

with open("prompt/personality_api_extract.prompt", 'r', encoding='utf-8') as f:
    personality_dimention_str = f.read()

class PersonalityTagListOutput(BaseModel):
    personality_tag_list: list[str] = Field(description="性格标签列表")

structured_llm = llm.with_structured_output(PersonalityTagListOutput, method="function_calling")

import json
def personality_tag_list_node(description: str) -> dict:

    messages = [
        {"role": "system", "content": personality_dimention_str},
        {"role": "user", "content": f"请处理以下 JSON 数据：\n{description}"},
    ]

    result = structured_llm.invoke(messages)
    return result.personality_tag_list