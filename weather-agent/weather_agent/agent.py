from datetime import date

from langchain.agents import load_tools, OpenAIFunctionsAgent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from langchain.agents import tool


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

Remember, use `google_search` for all weather-related queries to get the most accurate and up-to-date information. You do not need to limit your search to just the examples provided; use the tool as needed to answer the weather-related questions accurately.
When providing weather information, make sure to:
1. Convert temperature to Celsius and round to one decimal place.
2. Provide the Sensory temperature in Celsius, also rounded to one decimal place.
3. Convert wind speed from miles per hour (mph) to meters per second (m/s) and indicate the wind force level in parentheses using the Beaufort scale.
4. Provide a concise response including the current weather, temperature, sensory temperature, wind speed, and corresponding wind force level.

For example:

<question>What is the date today?</question>
<logic>Use the `time` tool to get today's date.</logic>
"""

prompt = ChatPromptTemplate.from_messages(  # 创建prompt
    [
        ("system", TEMPLATE),  # 模板
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # 代理的scratchpad
        ("human", "{input}"),  # 输入
    ]
)

# 加载 OpenAI 模型
llm = ChatOpenAI(temperature=0, max_tokens=2048, model="gpt-3.5-turbo")

# 加载 serpapi 工具
tools = load_tools(["serpapi"])

# 创建agent
agent = OpenAIFunctionsAgent(
    llm=llm, prompt=prompt, tools=tools + [time]
)

# 创建agent_executor
agent_executor = AgentExecutor(  # 创建agent_executor
    agent=agent, tools=tools + [time], max_iterations=5, early_stopping_method="generate"
    # 指定agent、tools、max_iterations、early_stopping_method
) | (lambda x: x["output"])


class AgentInputs(BaseModel):
    input: str


agent_executor = agent_executor.with_types(input_type=AgentInputs)
# agent_executor.invoke({"input": "武汉的天气怎么样"})
