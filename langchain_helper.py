from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0.6)



embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],model_name="hkunlp/instructor-xl")
# embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


vectordb_file_path="faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='ggv_faqs.csv', source_column='Question')
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)

    loader1 = WebBaseLoader(["https://josaa.admissions.nic.in/applicant/seatmatrix/InstProfile.aspx?enc=nsU5HEkjvt/OC38zhsZ0ytGD/1D+L0n4WyLfOwyFk4="])
    data = loader1.load()
    docs1 = text_splitter.split_documents(data)
    vectordb.add_documents(docs1)
    
    
    vectordb.save_local(vectordb_file_path)
    print("Success")

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path,embeddings,allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """You are a helpful college office assistant who answers to the query of students regarding courses, fess etc.
    Use the following pieces of context to answer the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    CONTEXT:{context}

    QUESTION:{question}"""

    PROMPT = PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True,
                            chain_type_kwargs={"prompt":PROMPT})
    
    return chain

def load_directory_to_vector_store(dir_path):
    loader = DirectoryLoader(dir_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    vectordb = FAISS.load_local(vectordb_file_path,embeddings,allow_dangerous_deserialization=True)
    vectordb.add_documents(texts)
    vectordb.save_local(vectordb_file_path)
    print("Scussfully loaded and saved")

if __name__ == "__main__":
    # create_vector_db()
    chain = get_qa_chain()

    # print(chain.invoke("Where is SoS E&T ?"))

    # load_directory_to_vector_store("./data/")