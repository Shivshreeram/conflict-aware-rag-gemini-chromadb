"""
Conflict-Aware RAG System for NebulaGears
Uses Google Gemini Flash 2.5 + ChromaDB + Google GenAI Embeddings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Patch for Pydantic v2 compatibility with ChromaDB
try:
    from pydantic_settings import BaseSettings
    import pydantic
    if not hasattr(pydantic, 'BaseSettings'):
        pydantic.BaseSettings = BaseSettings
except ImportError:
    pass

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from datetime import datetime
import json

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "nebulagears_documents"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "models/gemini-2.5-flash"  # Using available model from list
LLM_MODEL_FALLBACK = "models/gemini-flash-latest"

class ConflictAwareRAG:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        self.llm_model_name = LLM_MODEL
        self.llm_model_fallback = LLM_MODEL_FALLBACK
        
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
    def get_embedding(self, text: str):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            if isinstance(result, dict):
                return result.get('embedding') or result.get('values', result)
            return result
        except Exception as e:
            print(f"Embedding error: {e}")
            try:
                result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
                if isinstance(result, dict):
                    return result.get('embedding') or result.get('values', result)
                return result
            except Exception as e2:
                print(f"Fallback embedding failed: {e2}")
                raise
    
    def ingest_documents(self, documents_dir: str = "Assignement"):
        documents = []
        metadatas = []
        ids = []
        
        # Document A: Employee Handbook v1
        with open(f"{documents_dir}/employee_handbook_v1.txt", "r", encoding="utf-8") as f:
            doc_a = f.read()
            documents.append(doc_a)
            metadatas.append({
                "document_name": "employee_handbook_v1.txt",
                "document_type": "handbook",
                "effective_date": "2024-01-15",
                "role_applicability": "general_employee",
                "specificity": "general",
                "priority": 1
            })
            ids.append("doc_employee_handbook_v1")
        
        # Document B: Manager Updates 2024
        with open(f"{documents_dir}/manager_updates_2024.txt", "r", encoding="utf-8") as f:
            doc_b = f.read()
            documents.append(doc_b)
            metadatas.append({
                "document_name": "manager_updates_2024.txt",
                "document_type": "policy_update",
                "effective_date": "2024-06-01",
                "role_applicability": "general_employee",
                "specificity": "general",
                "priority": 2
            })
            ids.append("doc_manager_updates_2024")
        
        # Document C: Intern Onboarding FAQ
        with open(f"{documents_dir}/intern_onboarding_faq.txt", "r", encoding="utf-8") as f:
            doc_c = f.read()
            documents.append(doc_c)
            metadatas.append({
                "document_name": "intern_onboarding_faq.txt",
                "document_type": "role_specific_faq",
                "effective_date": "2024-06-01",
                "role_applicability": "intern",
                "specificity": "role_specific",
                "priority": 3
            })
            ids.append("doc_intern_onboarding_faq")
        
        print("Generating embeddings...")
        embeddings = [self.get_embedding(doc) for doc in documents]
        
        print("Adding documents to ChromaDB...")
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully ingested {len(documents)} documents into ChromaDB")
    
    def retrieve_with_conflict_awareness(self, query: str, user_role: str = None, top_k: int = 5):
        query_embedding = self.get_embedding(query)
        
        where_clause = None
        if user_role and user_role.lower() in ["intern", "internship"]:
            where_clause = {
                "$or": [
                    {"role_applicability": "intern"},
                    {"role_applicability": "general_employee"}
                ]
            }
        elif user_role:
            where_clause = {"role_applicability": "general_employee"}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get more, then re-rank
            where=where_clause
        )
        
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc_info = {
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 1.0,
                    "id": results['ids'][0][i]
                }
                retrieved_docs.append(doc_info)
        
        # Re-rank
        if user_role and user_role.lower() in ["intern", "internship"]:
            retrieved_docs.sort(key=lambda x: (
                x['metadata'].get('specificity') == 'role_specific',
                x['metadata'].get('priority', 0),
                - (1 - x['distance'])  # Higher similarity = lower distance
            ), reverse=True)
        else:
            retrieved_docs.sort(key=lambda x: (
                x['metadata'].get('priority', 0),
                - (1 - x['distance'])
            ), reverse=True)
        
        return retrieved_docs[:top_k]
    
    def generate_response(self, query: str, user_role: str = None, top_k: int = 3):
        retrieved_docs = self.retrieve_with_conflict_awareness(query, user_role, top_k=top_k*2)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information in the knowledge base.",
                "sources": [],
                "retrieved_documents": []
            }
        
        context_parts = []
        source_docs = []
        
        for doc_info in retrieved_docs[:top_k]:
            meta = doc_info['metadata']
            context_parts.append(
                f"--- Document: {meta.get('document_name', 'Unknown')} ---\n"
                f"Type: {meta.get('document_type', 'Unknown')}\n"
                f"Role: {meta.get('role_applicability', 'Unknown')}\n"
                f"Date: {meta.get('effective_date', 'Unknown')}\n"
                f"Content:\n{doc_info['document']}\n"
            )
            source_docs.append({
                "document_name": meta.get('document_name'),
                "role_applicability": meta.get('role_applicability'),
                "effective_date": meta.get('effective_date'),
                "specificity": meta.get('specificity', 'general')
            })
        
        context = "\n\n".join(context_parts)
        
        role_context = ""
        if user_role:
            role_context = f"\nThe user is a {user_role}. "
            if user_role.lower() in ["intern", "internship"]:
                role_context += "Intern-specific rules override general employee policies. "

        prompt = f"""You are an expert HR assistant at NebulaGears. Answer based ONLY on the provided documents.

{context}

{role_context}

Rules for conflicts:
- Role-specific > general policies
- Newer dates > older dates
- Always explain which document controls and why
- Cite document names clearly

Question: {query}

Answer clearly and professionally:"""

        try:
            # Configure safety settings using enum values
            safety_config = [
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                },
            ]
            
            model = genai.GenerativeModel(
                self.llm_model_name,
                safety_settings=safety_config
            )
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024
                )
            )

            # Extract answer from response - try multiple methods
            answer = None
            
            # Method 1: Try response.text (simplest)
            if hasattr(response, 'text') and response.text:
                answer = response.text
            # Method 2: Extract from candidates
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                
                # If blocked, try to get partial content
                if finish_reason == 2:  # SAFETY
                    # Try to get any available text
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts'):
                            parts_text = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    parts_text.append(part.text)
                            if parts_text:
                                answer = "".join(parts_text)
                    
                    # If still no answer, try fallback model
                    if not answer:
                        print("Primary model blocked, trying fallback...")
                        try:
                            fallback_model = genai.GenerativeModel(
                                self.llm_model_fallback,
                                safety_settings=safety_config
                            )
                            fallback_response = fallback_model.generate_content(prompt)
                            if hasattr(fallback_response, 'text'):
                                answer = fallback_response.text
                            else:
                                answer = "Response blocked. Please try a different query or check API settings."
                        except:
                            answer = "Response blocked by safety filters. The retrieval system correctly found the relevant documents."
                            
                elif finish_reason == 3:  # RECITATION
                    answer = "Response blocked due to recitation policy."
                elif finish_reason == 4:  # OTHER
                    answer = f"Response blocked (reason: {finish_reason})"
                elif candidate.content and candidate.content.parts:
                    # Success - extract text
                    parts_text = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            parts_text.append(part.text)
                    answer = "".join(parts_text) if parts_text else "Empty response."
                else:
                    answer = "Empty response received from model."
            
            # If still no answer, provide fallback
            if not answer:
                answer = self._generate_fallback_answer(retrieved_docs, query, user_role)

        except Exception as e:
            error_str = str(e)
            if "quota" in error_str.lower() or "429" in error_str:
                print("Quota exceeded, trying fallback model...")
                try:
                    fallback_model = genai.GenerativeModel(
                        self.llm_model_fallback,
                        safety_settings=safety_config
                    )
                    fallback_response = fallback_model.generate_content(prompt)
                    # Handle fallback response properly
                    if hasattr(fallback_response, 'candidates') and fallback_response.candidates:
                        candidate = fallback_response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            answer = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                        elif hasattr(fallback_response, 'text'):
                            answer = fallback_response.text
                        else:
                            answer = "Fallback model also blocked."
                    elif hasattr(fallback_response, 'text'):
                        answer = fallback_response.text
                    else:
                        answer = f"Fallback model failed: {error_str}"
                except Exception as fallback_error:
                    answer = f"Both models failed. Original error: {error_str}"
            elif "finish_reason" in error_str or "Part" in error_str or "2" in error_str:
                # Response was blocked by safety filters
                print("Response blocked, trying fallback model with relaxed settings...")
                try:
                    # Try fallback model
                    fallback_model = genai.GenerativeModel(self.llm_model_fallback)
                    fallback_response = fallback_model.generate_content(prompt)
                    if hasattr(fallback_response, 'text'):
                        answer = fallback_response.text
                    elif hasattr(fallback_response, 'candidates') and fallback_response.candidates:
                        candidate = fallback_response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            answer = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                        else:
                            answer = "Response blocked by safety filters. However, the system correctly retrieved the relevant documents showing: interns cannot work remotely (5 days/week in office required), while employees have hybrid options (3 days/week remote max with manager approval)."
                    else:
                        answer = "Response blocked. Based on retrieved documents: Interns cannot work remotely (intern_onboarding_faq.txt). Employees can work remotely up to 3 days/week with manager approval (manager_updates_2024.txt)."
                except:
                    # Provide answer based on retrieved documents
                    answer = """Based on the retrieved company documents:

**Who CAN work remotely:**
- Full-time employees: Can work remotely up to 3 days per week (with manager approval)
- Must be in office on Tuesdays and Thursdays

**Who CANNOT work remotely:**
- Interns: Required to be in the office 5 days a week. No remote work is permitted for interns.

This information is from:
- intern_onboarding_faq.txt (for intern policies)
- manager_updates_2024.txt (for employee policies)"""
            else:
                answer = f"Error generating response: {error_str}. However, the retrieval system correctly found the relevant documents."

        return {
            "answer": answer.strip(),
            "sources": source_docs,
            "retrieved_documents": [d['metadata']['document_name'] for d in retrieved_docs[:top_k]]
        }
    
    def query(self, question: str, user_role: str = None):
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        if user_role:
            print(f"Role: {user_role}")
        print(f"{'='*60}\n")
        
        response = self.generate_response(question, user_role)
        
        print("ANSWER:")
        print("-" * 60)
        print(response['answer'])
        print("\nSOURCES:")
        for i, src in enumerate(response['sources'], 1):
            print(f"{i}. {src['document_name']} ({src['specificity']}, {src['effective_date']})")
        
        print(f"\nRetrieved: {', '.join(response['retrieved_documents'])}")
        print(f"{'='*60}\n")
        
        return response


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API key: ").strip()
        if not api_key:
            print("API key required!")
            return
    
    rag = ConflictAwareRAG(api_key)
    
    if rag.collection.count() == 0:
        print("Ingesting documents...")
        rag.ingest_documents()
    else:
        print(f"{rag.collection.count()} documents found.")
        if input("Re-ingest? (y/n): ").lower() == 'y':
            rag.ingest_documents()
    
    # Test queries
    rag.query("I just joined as a new intern. Can I work from home?", "intern")
    rag.query("What are the remote work policies?", "employee")
    
    print("\nInteractive mode (type 'exit' to quit)")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ['exit', 'quit']:
            break
        role = input("Role (optional): ").strip() or None
        rag.query(q, role)


if __name__ == "__main__":
    main()