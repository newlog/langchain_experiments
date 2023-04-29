from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile


def start():
    summary_template = """
    Given the LinkedIn {information} about a person from I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # Temperature defines how creative the LLM will be. 0 means no creative.
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    linkedin_profile_url = linkedin_lookup_agent(name="Albert Lopez Fernandez")
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url)
    # The "information" named parameter has this name because we have named
    # the input_variable of the Prompt Template as "information". aka, dark magic.
    output = chain.run(information=linkedin_data)
    print(output)


if __name__ == "__main__":
    start()
