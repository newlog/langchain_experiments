import os
import requests


def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
    res = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header
    )
    return __clean_linkedin_profile_data(res.json())


def __clean_linkedin_profile_data(linkedin_data: dict) -> dict:
    data = {
        k: v
        for k, v in linkedin_data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    for group in data.get("groups", []):
        group.pop("profile_pic_url")
    return data
