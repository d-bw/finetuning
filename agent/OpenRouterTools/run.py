# run example

from langchain.agents import AgentExecutor

from llm import OpenRouterTools
from tools import SearchTool, DrawTool, AudioTool
from agent import IntentAgent
import os

os.environ['OPENROUTER_API_KEY']="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"
# google search api ley
GOOGLE_API_KEY = "AIzaSyAaaVpsNzZpUMCp5lMi8LcHoJi_r_-Mmg0"
GOOGLE_CSE_ID = "91e8826cbe43142ed"

# baidu translate api key
BAIDU_APPID = "20220306001112842"
BAIDU_APPKEY = "3kls_MFDSXSivPbfqHk5"


llm = OpenRouterTools(model_path="gpt-3.5")
llm.load_model("gpt-3.5")

tools = [SearchTool(llm=llm, google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID),
         DrawTool(baidu_appid=BAIDU_APPID, baidu_appkey=BAIDU_APPKEY),
         AudioTool()]

agent = IntentAgent(tools=tools, llm=llm)
agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)
agent_exec.run("美国第一任总统是谁？")