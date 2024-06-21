import os
import openai
from langchain.agents import load_tools, OpenAIFunctionsAgent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain_community.utilities import SerpAPIWrapper
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

# 定义 agent 的模板
TEMPLATE = """You are an AI assistant that provides real-time weather information using Google Search via SerpAPI.
You have a tool called `google_search` that allows you to perform a Google search for the current weather in a specified location.

The typical way to search for the weather is to query something like "current weather in <location>".

For example:

<question>What is the weather like in New York today?</question>
<logic>Use `google_search` with the query "current weather in New York".</logic>

<question>Is it raining in Tokyo?</question>
<logic>Use `google_search` with the query "current weather in Tokyo".</logic>

You can use the `time` tool to get the current date if necessary for your query.

For example:

<question>What is the date today?</question>
<logic>Use the `time` tool to get today's date.</logic>

Remember, use `google_search` for all weather-related queries to get the most accurate and up-to-date information. You do not need to limit your search to just the examples provided; use the tool as needed to answer the weather-related questions accurately.
"""

prompt = ChatPromptTemplate.from_messages(  # 创建prompt
    [
        ("system", TEMPLATE),  # 模板
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # 代理的scratchpad
        ("human", "{input}"),  # 输入
    ]
)

# 创建agent
agent = OpenAIFunctionsAgent(
    llm=llm, prompt=prompt, tools=tools + [time]
)

# 创建agent_executor
agent_executor = AgentExecutor(  # 创建agent_executor
    agent=agent, tools=tools + [time], max_iterations=5, early_stopping_method="generate"
    # 指定agent、tools、max_iterations、early_stopping_method
) | (lambda x: x["output"])

print(f'agent_executor: {agent_executor}')


class AgentInputs(BaseModel):
    input: str


agent_executor = agent_executor.with_types(input_type=AgentInputs)
# agent_executor.invoke({"input": "今天武汉的天气怎么样"})

# # 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
# agent = initialize_agent(tools + [time], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
#
# # 运行 agent
# print(agent.run("今天武汉的天气怎么样"))
