"""
DATE: 2023/5/28
AUTHOR: ZLYANG
CONTACT: zhlyang95@hotmail.com
"""

# run example

from langchain.agents import AgentExecutor

from llm import ChatGLM
from tools import SearchTool, DrawTool, AudioTool
from agent import IntentAgent


# google search api ley
GOOGLE_API_KEY = "AIzaSyAaaVpsNzZpUMCp5lMi8LcHoJi_r_-Mmg0"
GOOGLE_CSE_ID = "91e8826cbe43142ed"

# baidu translate api key
BAIDU_APPID = "20220306001112842"
BAIDU_APPKEY = "3kls_MFDSXSivPbfqHk5"


llm = ChatGLM(model_path="THUDM/chatglm-6b")
llm.load_model()

tools = [SearchTool(llm=llm, google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID),
         DrawTool(baidu_appid=BAIDU_APPID, baidu_appkey=BAIDU_APPKEY),
         AudioTool()]

agent = IntentAgent(tools=tools, llm=llm)
agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)
agent_exec.run("北京工业大学现任校长是谁？")