import datetime

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

json_parser = JsonOutputToolsParser(return_id=True)

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher. 
        Current time: {time} 
        
        1. {first_instruction}
        2. Reflect and critique your answer. Be sever to maximize improvement. 
        3. MUST Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_respoonder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revision_instructions="""Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""
revisor_prompt_template=actor_prompt_template.partial(first_instruction=revision_instructions) 

revisor_chain=revisor_prompt_template | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer") # tool_choice="ReviseAnswer" enforces the schema of the pydantic object ReviseAnswer class

if __name__ == "__main__":
    human_message = HumanMessage(
        content="""Write about AI-Powered SOC/autonomous soc problem domain, 
                               list startups that do that and raised capital."""
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion])
        | pydantic_parser
    )
    
    response=chain.invoke(input={"messages": [human_message]}) 
    
    print(response)
