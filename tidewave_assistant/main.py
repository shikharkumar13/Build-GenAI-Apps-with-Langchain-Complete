import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LANGSMITH_TRACING", "true")

from assistant import build_assistant
from knowledge_base import build_retriever
from triage import build_triage


def main() -> None:
    agent = build_assistant(retriever=build_retriever(persist_directory=".chroma"))
    triage = build_triage()
    config = {"configurable": {"thread_id": "customer-sam"}}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        classification = triage.invoke({"message": user_input})
        if classification.priority == "high":
            print(f"[escalation] {classification.category}, high priority: {classification.summary}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print("Assistant:", result["messages"][-1].text)


if __name__ == "__main__":
    main()
