from macos_use.llms.anthropic import ChatAnthropic
from macos_use.agent import Agent,Browser
from dotenv import load_dotenv

load_dotenv()

def main():
    llm=ChatAnthropic(model="claude-haiku-4-5")
    agent = Agent(llm=llm,log_to_file=True)
    query=input("Enter your query: ")
    result=agent.invoke(query)
    print(result)

if __name__ == "__main__":
    main()