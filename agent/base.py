
from typing import TypedDict, List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

api_key = ""   # deepseek
base_url = "https://api.deepseek.com"  #
model = "deepseek-flash"

llm = ChatOpenAI(
    model = model, openai_api_key = api_key, openai_api_base = base_url,
    temperature=0.1, max_tokens=2048,
    # extra_body={"enable_thinking": False}  # qwen
    extra_body = {"thinking": {"type": "disabled"}}# deepseek
)
