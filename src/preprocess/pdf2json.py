# import re
# from pdfminer.high_level import extract_pages, extract_text


file = "/storage/usmanova/judgeclaim/reports/original/2022 Microsoft Environmental Sustainability Report.pdf"

# text = extract_text(file, page_numbers=[4,6])

# with open("/storage/usmanova/judgeclaim/reports/processed/2022 Microsoft Environmental Sustainability Report.txt", "w") as f:
#     f.write(text)



from langchain_community.document_loaders import PyPDFLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

from typing import List

API_KEY = ""

llm = ChatOpenAI(api_key=API_KEY)

class Document(BaseModel):
    title: str = Field(description="Report title")
    author: str = Field(description="Report author")
    overview: str = Field(description="Company overview")
    sustainability: str = Field(description="Company sustinability and environment protection initiatives")
    keywords: List[str] = Field(description="Kewords used")

parser = JsonOutputParser(pydantic_object=Document)

def load_pdf():
    loader = PyPDFLoader(file)
    pages = loader.load()

    return pages

prompt = PromptTemplate(
    template="Extract the information as specified.\n{format_instructions}\n{cotext}",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

pages = load_pdf()

chain = prompt | llm | parser # Langchain Expression Language: format to use to build chains

response = chain.invoke({
    "context": pages
})  # Sending pages as chunks of pdf content using the given format we specified

print(response)
