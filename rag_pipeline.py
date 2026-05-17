import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

# Initialize Pinecone
pc_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "portfolio-chatbot")

index = None
if pc_api_key:
    try:
        pc = Pinecone(api_key=pc_api_key)
        index = pc.Index(index_name)
    except Exception as e:
        print(f"Warning: Could not connect to Pinecone index '{index_name}': {e}")

def get_embedding(text: str) -> list[float]:
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-2",
            content=text,
            task_type="retrieval_query",
            output_dimensionality=1024
        )
        return result['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

def retrieve_context(query: str, top_k: int = 3) -> str:
    if not index:
        return "No external context available. Pinecone index is not initialized."
    
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return ""
            
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        contexts = []
        for match in results.matches:
            if 'text' in match.metadata:
                contexts.append(match.metadata['text'])
                
        return "\n\n---\n\n".join(contexts)
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

# Define user personas and their suggested questions
PERSONAS = {
    "HR": {
        "behavior": "Formal, concise, highlights key achievements, quantified impact, and professional readiness.",
        "questions": [
            "What is Lalitendra's current notice period?",
            "What roles is he open to?",
            "What are his salary expectations?",
            "Can you share his resume PDF?"
        ]
    },
    "Interviewer": {
        "behavior": "Technical depth, honest about strengths and learning areas, gives code examples or architectural explanations if relevant.",
        "questions": [
            "Explain your experience with MERN stack",
            "How did you handle state management in your projects?",
            "What's your experience with cloud platforms (AWS/Azure)?",
            "Walk me through your most challenging project"
        ]
    },
    "Student": {
        "behavior": "Encouraging, mentor-like, shares personal journey, advice, and learning resources.",
        "questions": [
            "How did you start your dev career?",
            "What resources do you recommend for learning React?",
            "Do you offer mentorship?",
            "What skills should I focus on in 2025?"
        ]
    },
    "Freelancer": {
        "behavior": "Business-focused, highlights deliverables, availability, project process, and professionalism.",
        "questions": [
            "What services do you offer?",
            "What's your typical project timeline?",
            "Can I see examples of your past client work?",
            "How do we start working together?"
        ]
    }
}

def generate_response(message: str, user_type: str, chat_history: list[dict]) -> dict:
    context = retrieve_context(message)
    
    persona = PERSONAS.get(user_type, PERSONAS["Student"])
    behavior = persona["behavior"]
    suggested_questions = persona["questions"]
    
    system_prompt = f"""
You are Lalitendra Swamy's AI assistant for his 3D Portfolio.
You are currently talking to a visitor whose profile is: {user_type}.
Your behavior should be: {behavior}

Use the following information from Lalitendra's knowledge base to answer the user's question.
If the answer is not in the context, politely say you don't have that specific information but offer something relevant.
Do not make up facts about Lalitendra.

Knowledge Base Context:
{context}

Respond directly to the user's latest message. Ensure the response format is clean and readable.
Keep responses concise, usually 2-4 short paragraphs maximum.
"""
    
    history_text = ""
    if chat_history:
        for msg in chat_history[-5:]: # Include last 5 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role.capitalize()}: {content}\n"
            
    full_prompt = f"{system_prompt}\n\nChat History:\n{history_text}\nUser: {message}\nAssistant:"
    
    try:
        response = model.generate_content(full_prompt)
        answer = response.text
    except Exception as e:
        answer = f"I'm sorry, I encountered an error while trying to process your request. ({str(e)})"
    
    return {
        "answer": answer,
        "suggested_questions": suggested_questions
    }
