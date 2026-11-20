from agent.base import *

with open("prompt/chat.prompt", 'r', encoding='utf-8') as f:
    chat_template_str = f.read()

class ChatTextOutput(BaseModel):
    text: str = Field(description="对话文本")

structured_llm = llm.with_structured_output(ChatTextOutput, method="function_calling")
# print("chat_template_str", chat_template_str)
import json
def chat_node(hist_chat_list: list, info_questioned_dic:dict, info_answered_dic:dict) -> dict:
    input_str = json.dumps({
        "hist_chat_list": hist_chat_list,
        "info_questioned_dic": info_questioned_dic,
        "info_answered_dic": info_answered_dic},  ensure_ascii=False)
    # system_content = chat_template_str.replace("{input_json}", input_str)

    messages = [
        {"role": "system", "content": chat_template_str},
        {"role": "user", "content": f"请处理以下 JSON 数据：\n{input_str}"},
    ]

    result = structured_llm.invoke(messages)
    return result.text