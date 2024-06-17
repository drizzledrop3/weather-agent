import os
import openai
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.agents import tool
from datetime import date

# 需要在环境中配置 key 如果是中转key 需要配置中转的url
# # 设置 OpenAI API 密钥
# openai.api_key = os.getenv('OPENAI_API_KEY')
#
# # 设置中转 API 的基本 URL
# openai.api_base = os.getenv('OPENAI_BASE_URL')


@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())


# 加载 OpenAI 模型
llm = OpenAI(temperature=0, max_tokens=2048)

# 加载 serpapi 工具
tools = load_tools(["serpapi"])

# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
agent = initialize_agent(tools + [time], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# 运行 agent
print(agent.run("今天武汉的天气怎么样"))
