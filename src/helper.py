import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import TokenTextSplitter  # Correct import for TokenTextSplitter





prompt_template = """"
You are an expert at creating question based on DBMS material and documentation.
Your goal is to prepare a coder or programer for their exam and coding tests.
You do this by asking questions about text below:
---------
{text}
---------
create ten question that will prepare the coders or programers for their tests.
Make sure not to lose any important information.

QUESTION:
"""


refine_template = ("""
        You are an expert at creating question based on DBMS material and documentation.
        Your goal is to prepare a coder or programer for their exam and coding tests.
        We have received some practical questions to a certain extent : {existing_answer}.
        we have the option to refine the existing question or add new ones.
        {text}
        prepare ten question 
        Given the new context, refine the original question in English.
        If the content is not helpful, please provide the original question.
        QUESTION:
""")


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def file_processing(file_path):
    # Load the PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Combine all page contents
    question_gen = ""
    for page in data:
        question_gen += page.page_content

    # Split text into chunks for question generation
    splitter_ques_gen = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=text) for text in chunks_ques_gen]

    # Split documents further for answer generation
    splitter_ans_gen = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    document_ans_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_ans_gen


def llm_pipeline(file_path):
    # Process the file to generate documents for question and answer generation
    document_ques_gen, document_answer_gen = file_processing(file_path)

    # Define the LLM for question generation
    llm_ques_gen_pipeline = ChatOpenAI(
        temperature=0.3,
        model="gpt-3.5-turbo"
    )

    # Define prompt templates
    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template,
        input_variables=["text"]
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        template=refine_template,
        input_variables=["existing_answer", "text"]
    )

    # Load the question generation chain
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )

    # Create a vector store from the documents for answer generation
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # Define the LLM for answer generation
    llm_answer_gen = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo"
    )

    # Example question generation
    ques = "What is the primary focus of DBMS?\nExplain normalization.\nWhat are the ACID properties?"

    # Filter out the questions from the list
    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    # Create the RetrievalQA chain
    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    return answer_generation_chain, filtered_ques_list