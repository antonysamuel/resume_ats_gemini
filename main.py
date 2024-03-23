import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from google.ai import generativelanguage as glm

load_dotenv('.env')


genai.configure(api_key= os.environ.get("GOOGLE_API_KEY"))


#define  function template
resume_template = glm.Tool(
    function_declarations = [
        glm.FunctionDeclaration(
            name = 'resume_parse',
            description = 'Parse the given resume and output name, email, phone, education, work_experience, skills, and others.',
            parameters = glm.Schema(
                type = glm.Type.OBJECT,
                properties = {
                    'name' : glm.Schema(type = glm.Type.STRING),
                    'email' : glm.Schema(type = glm.Type.STRING),
                    'phone' : glm.Schema(type = glm.Type.STRING),
                    'education' : glm.Schema(type = glm.Type.STRING),
                    'work experience' : glm.Schema(type = glm.Type.STRING),
                    'skills' : glm.Schema(type = glm.Type.STRING),
                },
                required = ['name']
            )
        )
    ]
)


#define model
model = genai.GenerativeModel(model_name= 'gemini-1.0-pro', tools= resume_template)




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

    Parse the given resume



"""

chat = model.start_chat()
response = chat.send_message(prompt)

print(response.candidates[0].content.parts[0].function_call)