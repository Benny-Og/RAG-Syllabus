<h1>Syllabus RAG Assistant (LLM + Vector Search)</h1>

<h2>Description</h2>

This project is a Retrieval-Augmented Generation (RAG) system built to act as an academic advisor for querying course syllabi. The system ingests PDF syllabi, processes and chunks the text, and stores embeddings in a vector database for semantic search.  

When a user asks a question, the most relevant sections of the syllabus are retrieved using similarity search and passed into a local LLM (via Ollama) to generate a context-aware answer. If the answer is not present in the retrieved content, the model is instructed to explicitly state that the information cannot be found.

This project demonstrates end-to-end implementation of document parsing, embedding generation, vector storage, retrieval, and LLM-based question answering.

<br />

<h2>Core Features</h2>

- Loads multiple PDF syllabi from a local directory (`./syllabi`)
- Splits documents into overlapping text chunks for better retrieval
- Generates embeddings using HuggingFace Sentence Transformers
- Stores and persists embeddings in a Chroma vector database
- Performs semantic retrieval based on user queries
- Uses a local LLM (Ollama `phi3:mini`) for answer generation
- Interactive CLI for real-time question answering

<br />

<h2>Languages and Utilities Used</h2>

- <b>Python</b>  
- <b>LangChain</b> (Prompting, chaining, retrievers)  
- <b>Ollama (phi3:mini)</b> – Local LLM inference  
- <b>ChromaDB</b> – Vector database storage  
- <b>HuggingFace Embeddings</b> (`sentence-transformers/all-MiniLM-L6-v2`)  
- <b>PyPDFLoader</b> – PDF document ingestion  
- <b>RecursiveCharacterTextSplitter</b> – Text chunking strategy  
- <b>dotenv</b> – Environment variable management  

<br />

<h2>System Workflow</h2>

1. Load all PDFs from the `syllabi` folder  
2. Extract raw text from each document  
3. Split text into overlapping chunks  
4. Convert chunks into vector embeddings  
5. Store embeddings in a persistent Chroma database  
6. Accept user questions via CLI  
7. Retrieve top-k relevant chunks using similarity search  
8. Pass context + question into LLM prompt chain  
9. Generate and return response  

<br />

<h2>File Access</h2>

- <b>Run the main Python script to start the interactive Q&A system</b>  
- <b>Ensure PDF files are placed in the `./syllabi` directory before execution</b>  
- <b>Vector database is persisted locally in `./chroma_db`</b>  
