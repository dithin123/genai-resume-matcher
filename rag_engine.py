from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def generate_match_report(resume, jd):
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(resume + jd)

    db = FAISS.from_texts(docs, embeddings)
    context = db.similarity_search(jd, k=4)

    prompt = f"""
You are an expert ATS evaluator.

Resume:
{resume}

Job Description:
{jd}

Context from RAG:
{context}

Generate:
- Match percentage
- Missing skills
- Improved resume bullet points
- Summary why the candidate fits
- Final evaluation
"""

    response = llm.predict(prompt)
    return response
