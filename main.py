import os
import time
import ast
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from google.ai import generativelanguage as glm
import json

load_dotenv('.env')


genai.configure(api_key= os.environ.get("GOOGLE_API_KEY"))


#define  function template
resume_template = glm.Tool(
    function_declarations = [
        glm.FunctionDeclaration(
            name = 'resume_parse',
            description = 'Parse the given resume and output name, email, phone, education, work_experience, skills, and others',
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
model2 = genai.GenerativeModel(model_name= 'gemini-1.0-pro')





#Loading pdf 
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    page_contents = ' '
    
    for page in pages:
        page_contents += page.page_content 

    return page_contents


def parse_resume(file_path, job_title, job_description,skills):
    pdf_content = load_pdf(file_path)

    prompt_0 = f"""

         {pdf_content}
"

        Parse the given resume and output name, email, phone, education, work_experience, skills, and others

    """


    chat = model.start_chat()
    retry = 0
    while retry < 3:
        try:
            response = chat.send_message(prompt_0)
            retry = 3
        except:
            print("Retrying....!")
            retry += 1
            time.sleep(3)

    details = response.candidates[0].content.parts[0].function_call.args
    parser = {}
    for k,v in details.items():
        parser[k] = v


    
    prompt_1 = f"""
        Act as an efficient ATS System

        resume: ""{parser}""

        job_role: ""{job_title}""

        job_description: ""{job_description}""

        skills: ""{skills}""

        Parse the given resume and give out a score out of 100 for match found in the resume for the given job role in json format
        'summary': 'sumary of scoring',
        'score'  : 'score given on 100'
        
        

    """   
    retry = 0
    while retry < 5:
        try:
            result = model2.generate_content(prompt_1)
            parser['summary'] = result.text
            retry = 5
        except:
            print("Retrying....!")
            retry += 1
            time.sleep(3)
   
    return parser

if __name__ == '__main__':
    job_title = 'Data Science / AI / ML Consultant'
    job_description = ''' 
    As the Senior Data Scientist and reporting to the CTO, you will lead the India development team of the LPL’s space data analytics product Orbitfy, explore different use cases for the technology, and work closely with the CTO to strategize the product and company's roadmap for the next 3-4 years. 

Little Place Lab's data science team is the heart and the brain of our product development, offering space-derived analytics as the source of truth for key stakeholders so they can make critical and timely decisions.
You will work closely with engineering, products, and the CTO to apply sophisticated modeling techniques to a modern data stack and build the future of space-edge computing.
This is a remote position, with some occasional travel to locations within India.


Responsibilities

Team Leadership: Lead and mentor a diverse team of data scientists, engineers, and research analysts, fostering a collaborative and innovative working environment. Set clear team goals, delegate tasks effectively, and monitor progress.
Stakeholder Communication: Serve as the point of contact for internal and external stakeholders regarding work progress, technical challenges, and solutions. Engage in knowledge sharing and best practices within the team and across the company.
Cross-functional Collaboration: Work closely with the Geo-Information Science (GIS) engineering and software teams to integrate ML solutions into the broader data processing pipeline.
Recruitment and Onboarding: Assist in the recruitment, onboarding, and training of new team members, ensuring they align with the company's goals and culture.
Strategic Planning: Contribute to the strategic planning of AI/ML projects, identifying potential challenges and opportunities for growth within the team and broader organization.
Algorithm Development: Design, develop, and implement state-of-the-art computer vision algorithms for satellite imagery analysis, including object detection, image segmentation, and pattern recognition.
Data Handling: Manage and process large datasets of satellite imagery (such as optical, thermal, multispectral, hyperspectral, SAR, and telemetry), ensuring efficient data manipulation, preprocessing, and augmentation techniques.
Model Training and Optimization: Train, fine-tune, and optimize deep learning models for on-board deployment, focusing on maximizing efficiency and accuracy while minimizing computational and power requirements.
Innovation and Research: Stay abreast of the latest advancements in computer vision and machine learning, proposing and implementing innovative solutions to improve project outcomes.
Quality Assurance: Ensure the reliability and robustness of computer vision solutions through comprehensive testing, validation, and continuous improvement processes.
Documentation and Reporting: Maintain detailed documentation of algorithms, models, and processes; prepare reports and presentations for technical and non-technical audiences.


Qualifications

Master’s degree or Ph.D. in Computer Science, Data Science, Artificial Intelligence, or a related field.
Minimum of 8 years of experience in data science, with a strong portfolio of projects demonstrating expertise in computer vision, image processing, and analysis.
Proficiency in programming languages such as Python and C/C++ and libraries/frameworks like TensorFlow, PyTorch, OpenCV.
Experience in some of the following: Caffe, ONNX, CuBLAS, CUDA, OpenVINO, TensorflowLite, TensorRT, MKL DNN. 
Experience with deep learning architectures (CNNs, RNNs, GANs) specifically designed for computer vision tasks.
Knowledge of satellite imagery and remote sensing data processing is highly desirable.
Experience or understanding of FPGAs is highly desirable
Ability to work independently in a remote setting, with strong communication skills to collaborate effectively across time zones.
Creative problem-solving skills and a passion for innovation in the space technology domain.
    
    
    '''


    skills = ''' "python",
  "r",
  "sql",
  "scala",
  "java",
  "statistics",
  "probability",
  "machine learning",
  "deep learning",
  "data wrangling",
  "data cleaning",
  "data visualization",
  "matplotlib",
  "seaborn",
  "tableau",
  "power bi",
  "big data processing",
  "apache spark",
  "hadoop",
  "communication",
  "problem solving",
  "curiosity",
  "creativity",
  "collaboration"'''
    resumes = os.listdir('resumes/')
    for resume in resumes:
        print(f"------------{resume}----------------------")
        result = parse_resume(os.path.join('resumes',resume), job_title, job_description, skills)
        summary = result['summary']
        summary = summary.replace('```','').strip().replace('json','').strip()
        summary = json.loads(summary)
        score = summary["score"]
        print(f"{resume} : {score}")
        # break
        # print("----------------------------------------------")