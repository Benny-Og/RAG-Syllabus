from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

model = OllamaLLM(model="phi3:mini")

pdf_folder = Path('./syllabi')
for pdf in pdf_folder.glob('*.pdf'):
    print(f'PDF: ', pdf)

loader = PyPDFLoader(str(pdf))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(docs)

print(f'Total Chunks: ', len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

try:
    vector_store = Chroma.from_documents(
        documents = chunks,
        collection_name = 'my_vectors',
        embedding = embeddings,
        persist_directory = './chroma_db'
    )
    print(f'Successfully stored chunks to Chroma')
except:
    print(f'Could not store chunks to Chroma')

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

template = """
You are an expert academic advisor who is reading information about my syllabus.
If you cannot find the information I am requesting in the syllabus state that you could not find it.

Here is the syllabus: {syllabus}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print('\n\n------------------------')
    question = input("Ask your question (q to quit): ")
    if question == 'q':
        break

    docs = retriever.invoke(question)

    syllabus_text = '\n\n'.join(
        doc.page_content for doc in docs
    )

    result = chain.invoke({"syllabus": syllabus_text, "question": question})
    print(result)

