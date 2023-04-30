from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # Output Indicator = "Your answer should only contain the URL"
    template = """
    Given the full name {name_of_person} I want you to get me a link to their Linkedin profile page.
    Your answer should contain only the URL.
    """
    # The agent decides whether to use the Tool or not based on the description.
    tools_for_agent = [
        Tool(
            name="Crawl Google for Linkedin Profile Page",
            func=get_profile_url,
            description="Useful for when you need to get the Linkedin URL",
        )
    ]
    agent = initialize_agent(
        tools_for_agent, llm, AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
