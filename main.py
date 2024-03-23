import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv('.env')


genai.configure(api_key= os.environ.get("GOOGLE_API_KEY"))

#define model
model = genai.GenerativeModel(model_name= 'gemini-1.0-pro')

#define  function template


result = model.generate_content('Hi Gemini!!')

print(result.text)