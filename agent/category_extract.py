from agent.base import *

with open("prompt/category_extract.prompt", 'r', encoding='utf-8') as f:
    category_extract_str = f.read()

class CategoryExtractOutput(BaseModel):
    category_extract_dic: dict[str, bool] = Field(description="类别是否在对话中明确")

structured_llm = llm.with_structured_output(CategoryExtractOutput, method="function_calling")

import json
def category_extract_node(last_chat_list: list) -> dict:
    input_str = json.dumps({"hist_chat_list": last_chat_list},  ensure_ascii=False)

    messages = [
        {"role": "system", "content": category_extract_str},
        {"role": "user", "content": f"请处理以下 JSON 数据：\n{input_str}"},
    ]

    result = structured_llm.invoke(messages)
    print(result)
    #if result == None:
    #    return {}
    return result.category_extract_dic