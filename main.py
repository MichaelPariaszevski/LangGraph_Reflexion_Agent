from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from typing import List

from langchain_core.messages import BaseMessage, ToolMessage

from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_respoonder_chain
from tool_executor import execute_tools

MAX_ITERATIONS = 2

builder = MessageGraph()

builder.add_node(node="First_Draft", action=first_respoonder_chain)

builder.add_node(node="Execute_Tools", action=execute_tools)

builder.add_node(node="Revise", action=revisor_chain)

builder.add_edge(start_key="First_Draft", end_key="Execute_Tools")

builder.add_edge(start_key="Execute_Tools", end_key="Revise")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return "Execute_Tools"


builder.add_conditional_edges(source="Revise", path=event_loop)

builder.set_entry_point(key="First_Draft")

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph_illustration.png")

if __name__ == "__main__":
    print("Hello Reflexion Agent")
    response = graph.invoke(
        "Wrtie about AI-Powered SOC/autonomous soc problem domain, list startups that do that and raised capital."
    )
    print("-" * 50)
    print(response[-1].tool_calls[0]["args"]["answer"])
