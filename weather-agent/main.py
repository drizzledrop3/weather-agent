from weather_agent.agent import agent_executor

if __name__ == "__main__":
    # question = "What is the weather like in Wuhan?"
    question = "武汉的天气怎么样？"
    print(agent_executor.invoke({"input": question}))