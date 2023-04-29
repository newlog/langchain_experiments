from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class PersonIntel(BaseModel):
    summary: str = Field(description="Summary of the person")
    facts: list[str] = Field(description="Interesting facts about the person")

    def to_dict(self):
        return {"summary": self.summary, "facts": self.facts}


person_intel_parser = PydanticOutputParser(pydantic_object=PersonIntel)
