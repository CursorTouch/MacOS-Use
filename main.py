from macos_use.providers.nvidia import ChatNvidia
from macos_use.agent import Agent
from dotenv import load_dotenv

load_dotenv()

def main():
    # llm=ChatAnthropic(model="claude-haiku-4-5")
    # llm=ChatOllama(model="qwen3-vl:4b")
    # llm=ChatGroq(model="openai/gpt-oss-120b")
    llm=ChatNvidia(model="nvidia/nemotron-3-super-120b-a12b")
    # llm=ChatMistral(model="mistral-medium-3-5")

    agent = Agent(llm=llm,log_to_file=True,auto_minimize=True)
    query=input("Enter your query: ")
    result=agent.invoke(query)
    print(result)

if __name__ == "__main__":
    main()