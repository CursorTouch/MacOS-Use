from macos_use.llms.anthropic import ChatAnthropic
from macos_use.llms.ollama import ChatOllama
from macos_use.llms.groq import ChatGroq
from macos_use.agent import Agent
from dotenv import load_dotenv

load_dotenv()

def main():
    llm=ChatAnthropic(model="claude-haiku-4-5")
    # llm=ChatOllama(model="qwen3-vl:4b")
    # llm=ChatGroq(model="openai/gpt-oss-120b")
    agent = Agent(llm=llm,log_to_file=True)
    query=input("Enter your query: ")
    result=agent.invoke(query)
    print(result)

if __name__ == "__main__":
    main()