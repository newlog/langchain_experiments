from third_parties.linkedin import scrape_linkedin_profile


def test_linkedin():
    profile_url = "https://www.linkedin.com/in/albertlopezfernandez"
    linkedin_data = scrape_linkedin_profile(profile_url)
    assert len(linkedin_data) != 0
