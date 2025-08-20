from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
import uvicorn
from llm import LegalChatbot
from retriever import get_retriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Legal Chatbot API")

llm = LegalChatbot()
retriever = get_retriever()

template = """
You are a legal expert and assistant trained on Indian laws. Answer the question based on the provided context.
If unsure, respond with "I'm not sure, please consult a legal expert."
Always end with "This is a legal information service, not a substitute for professional legal advice."

Context: {context}

Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    output_key="result"
)

class DocumentMetadata(BaseModel):
    filename: str
    filepath: str
    page_number: int
    document_type: Optional[str] = "Legal"

class QuestionRequest(BaseModel):
    question: str

class ContractRequest(BaseModel):
    party1: str
    party2: str
    purpose: str
    duration: str
    date: str
    
@app.get("/metadata")
def get_metadata():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory="./embeddings",
        embedding_function=embedding_model
    )
    docs = vectordb.get(include=["metadatas"])
    return {
        "metadata": docs["metadatas"][:5]
    }

@app.post("/ask")
def ask_legal_questions(req: QuestionRequest):
    try:
        result = qa_chain.invoke({"query": req.question})
        sources = []
        for doc in result["source_documents"]:
            try:
                sources.append({
                    "content": doc.page_content[:200],
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page_number", 1),
                    "document_type": doc.metadata.get("document_type", "Legal"),
                    "time": doc.metadata.get("time", datetime.now().isoformat())
                })
            except KeyError as e:
                logger.warning(f"Missing metadata in document: {str(e)}")
                continue
        return {
            "response": result["result"],
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        # print(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail = f"Error processing your question: {str(e)}"
        )
    
@app.post("/summarize")
def summarize_documents(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name
        
        try:
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

            for i, doc in enumerate(docs):
                doc.metadata.update({
                    "filename": file.filename,
                    "filepath": temp_file_path,
                    "upload_time": datetime.now().isoformat(),
                    "document_type": "Legal",
                    "page_number": i + 1
                })

            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            vectordb = Chroma(
                persist_directory="./embeddings",
                embedding_function=embedding_model
            )

            vectordb.add_documents(docs)
            # vectordb.persist()

            full_text = "\n".join([doc.page_content for doc in docs])

            summary_prompt = f"""
            Provide a detailed summary by covering:
            1. Key points and arguments
            2. Important clauses and terms
            3. Top 3 risks or unusual clauses
            4. Governing law and jurisdiction

            \\n\\n{full_text[:8000]}
            Make it concise and easy to understand.
            """
            summary = llm.invoke(summary_prompt)

            return {
                "summary": summary,
                "metadata": docs[0].metadata
                # "metadata": {
                #     "filename": file.filename,
                #     "filepath": temp_file_path,
                #     "time": datetime.now().isoformat(),
                #     "document_type": "Legal",
                #     "page_count": len(docs)
                # }
            }
        finally:
            os.unlink(temp_file_path)  # Clean up temp file
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error summarizing the document"
        )

@app.post("/generate-contract")
def generate_contract(req: ContractRequest):
    try:
        contract_template=f"""
        Generate a comprehensive Non-Disclosure Agreement (NDA) between {req.party1} and {req.party2} with the following details:
        - Effective Date: {req.date}
        - Purpose: {req.purpose}
        - Duration: {req.duration}

        Include these standard clauses:
        1. Definition of Confidential Information
        2. Obligations of Receiving Party
        3. Exclusions from Confidential Information
        4. Term and Termination
        5. Return of Information
        6. Governing Law (India)
        7. Entire Agreement
        8. Signatures section

        Make it professionally formatted with clear sections and bullet points.
        """

        contract = llm.invoke(contract_template)
        return {"contract": contract}
    except Exception as e:
        logger.error(f"Error generating contract: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating the contract"
        )   

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)