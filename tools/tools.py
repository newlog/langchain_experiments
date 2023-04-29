from langchain import SerpAPIWrapper
from langchain.agents import tool


# This function will use the SerpAPI (Google Search API) via LangChain to do search lookups using Google.
#
# To remark here, the name of the parameter of this function is 'text' instead of e.g. 'name'. This is because
# the agent will call this function with strings that go beyond the name. To search if might use something like
# '<full_name> linkedin' or '<full_name> linkedin page'
@tool
def get_profile_url(text: str) -> str:
    """
    Searches for Linkedin Profile Page.

    :param text: Text that will be sent to the SerpAPI in order to search on Google.
    :return: Linkedin Profile Page.
    """
    search = SerpAPIWrapper()
    res = search.run(f"{text}")
    return res
