from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations with numbers"""
    print("Tool has been called")
    return f"The sum of {a} and {b} is {a + b}"

@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    print("Tool has been called")
    return f"Hello {name}, I hope you are well today!"


def main():
    model = ChatOllama(model="gemma4:e2b", temperature=0)

    tools = [calculator, say_hello]
    agent = create_agent(model=model, tools=tools)

    print ("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "quit":
            break

        print("\nAssistant: ", end="")
        result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]}
        )
        print(result["messages"][-1].content)

if __name__ == "__main__":
    main()