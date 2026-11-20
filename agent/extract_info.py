from typing import TypedDict, List, Tuple
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from functools import partial

from agent.base import llm

prompt_dic = {}
for category in ["personality", "hobby", "game"]:
    with open("prompt/%s_extract.prompt" % category, 'r', encoding='utf-8') as f:
        prompt_dic[category] = f.read()

class AgentState(TypedDict):
    personality_prefered: list[str]  # 偏好性格短语
    personality_not_prefered: list[str]  # 不喜欢的性格短语
    hobby_prefered: list[str]         # 偏好爱好标签列表
    game_prefered: list[str]          # 偏好游戏标签列表
    last_chat_list: list         # 最后一轮对话
    last_category_list: list          # 最后一轮信息类别

class PersonalOutput(BaseModel):
    personality_prefered: list[str] = Field(description="偏好的性格短语列表")
    personality_not_prefered: list[str] = Field(description="不喜欢的性格短语列表")

class TagListOutput(BaseModel):
    key: list[str] = Field(description="偏好的爱好、游戏短语列表")

def extract_node(state: AgentState, format_output: BaseModel, key: str) -> dict:
    key_name = {"personality": "性格", "hobby": "爱好", "game": "游戏"}.get(key, "")
    if key_name not in state.get("last_category_list", []):
        return {}

    structured_llm = llm.with_structured_output(format_output, method="function_calling")

    messages = [
        {"role": "system", "content": prompt_dic[key]},
        {"role": "user", "content": f"请处理以下对话：\n{str(state.get('last_chat_list', []))}"},
    ]

    # 调用结构化输出模型
    result = structured_llm.invoke(messages)
    # print(result)
    if key == "personality":
        return {
            "personality_prefered": state.get("personality_prefered", []) + result.personality_prefered,
            "personality_not_prefered": state.get("personality_not_prefered", []) + result.personality_not_prefered
        }
    else:
        key_stats = key + "_prefered"
        return { key_stats: state.get(key_stats, []) + result.key }

builder = StateGraph(AgentState)
builder.add_node("personality", partial(extract_node, format_output=PersonalOutput, key="personality"))
builder.add_node("hobby", partial(extract_node, format_output=TagListOutput, key="hobby"))
builder.add_node("game", partial(extract_node, format_output=TagListOutput, key="game"))
builder.set_entry_point("personality")
builder.add_edge("personality", "hobby")
builder.add_edge("hobby", "game")
builder.add_edge("game", END)
graph = builder.compile(checkpointer=MemorySaver())

def run_extract_workflow(last_chat_list:list, last_category_list:list, user_id: str):
    config = {"configurable": {"thread_id": user_id}}
    result = graph.invoke({
        "last_chat_list": last_chat_list, "last_category_list": last_category_list }, config)
    print(result)