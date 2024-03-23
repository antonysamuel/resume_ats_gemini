import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader

load_dotenv('.env')


genai.configure(api_key= os.environ.get("GOOGLE_API_KEY"))

#define model
model = genai.GenerativeModel(model_name= 'gemini-1.0-pro')

#define  function template






#Loading pdf 
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    page_contents = ' '
    
    for page in pages:
        page_contents += page.page_content 

    return page_contents


pdf_content = load_pdf('resumes/Data Engineer.pdf')

prompt = f"""

    {pdf_content}

    Summarize the given resume



"""
result = model.generate_content(prompt)

print(result.text)