import os
import time
import ast
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.evaluation import load_evaluator

from google.ai import generativelanguage as glm
import json

load_dotenv('.env')


genai.configure(api_key= os.environ.get("GOOGLE_API_KEY_1"))


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
embedding_model = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
evaluator = load_evaluator("embedding_distance", embeddings=embedding_model)



# Get the embeddings of each text and add to an embeddings column in the dataframe
def embed_fn(title, text):
  return genai.embed_content(model=model,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]


#Loading pdf 
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    page_contents = ' '
    
    for page in pages:
        page_contents += page.page_content 

    return page_contents


def parse_resume(file_path):
    pdf_content = load_pdf(file_path)

    prompt_0 = f"""

         {pdf_content}
"

        Parse the given resume and output name, email, phone, education, work_experience, skills, and others

    """


    chat = model.start_chat(enable_automatic_function_calling=True)
    retry = 0
    while retry < 3:
        try:
            response = chat.send_message(prompt_0)
            retry = 3
        except:
            print("Retrying....!")
            genai.configure(api_key= os.environ.get(f"GOOGLE_API_KEY_{retry%2}"))
            model_ = genai.GenerativeModel(model_name= 'gemini-1.0-pro', tools= resume_template)
            chat = model_.start_chat(enable_automatic_function_calling=True)

            retry += 1
            time.sleep(3)

    details = response.candidates[0].content.parts[0].function_call.args
    parser = {}
    for k,v in details.items():
        parser[k] = v
    return parser


def generate_resume_summary(candidate_details):
    prompt_1 = f'''Summarize the given resume in one paragraph dont use any headings like **summary** etc. Just output one paragraph
    {candidate_details}
    
    '''

    retry = 0
    while retry < 3:
        try:
            response = model2.generate_content(prompt_1)
            retry = 3
        except:
            print("<Summary> Retrying....!")
            retry += 1
            time.sleep(3)
    
    return response.text

def shortlist_resume(resume_folder, job):
    user_query = f'''Shortlist candidate who will be apt for the post of {job['title']}. More details regarding the job is given below:
    {job['description']}. The candidates having releavent experience will be prefered and also with releavent skills for the job which is {job['skills']} is to be 
    given importance.
    The candidate having releavent work experience for the job role {job['title']} will be an add-on.
    '''   

    resume_doc_list = []
    candidate_details = {}
    for resume in os.listdir(resume_folder):
        print(f'--------------{resume}------------')
        candidate_data_dict = parse_resume(os.path.join(resume_folder, resume))
        candidate_data_str = f'''The candidate have the following work experience : {candidate_data_dict['work_experience'] if 'work_experience' in candidate_data_dict else 'None'}. Skills of the candidate are :  {candidate_data_dict['skills'] if 'skills' in candidate_data_dict else 'None'}.
        The education of candidate are : {candidate_data_dict['education'] if 'education' in candidate_data_dict else 'None'}.  
        '''
        
        candidate_details[resume] = candidate_data_dict
        resume_doc_list.append(Document(page_content = candidate_data_str, metadata = dict(name = candidate_data_dict['name'], file_name = resume)))

    db = FAISS.from_documents(documents = resume_doc_list, embedding = embedding_model)
    result = db.similarity_search_with_score(user_query, k = 4)
    result_out = [dict(name = r[0].metadata['name'], file_name = r[0].metadata['file_name'], summary = generate_resume_summary(r[0].page_content), score = str(1 - r[1])) for r in result]
    for idx, r_out in enumerate(result_out):
        try:
            skills_ = candidate_details[r_out['file_name']]['skills']
            skill_score = evaluator.evaluate_strings(prediction = skills_, reference = job['skills'])
            result_out[idx]['skill_score'] = str(1 - skill_score['score'])
        except:
            print("Skills not found for ",r_out['file_name'])


    return result_out




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
    
    job_dict = dict(title = job_title, description = job_description, skills = skills)
    result = shortlist_resume('test_resume', job_dict)
    print(result)