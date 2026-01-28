from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Chat API with OpenAI GPT-4o-mini")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Request model for chat endpoint
class ChatRequest(BaseModel):
    question: str
    conversation_history: list = []


# Response model for chat endpoint
class ChatResponse(BaseModel):
    answer: str
    conversation_history: list


@app.get("/")
def read_root():
    return {"message": "Chat API is running. Use POST /chat to ask questions."}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that takes a user question and returns AI response using GPT-4o-mini
    """
    try:
        # Check if API key is set
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in .env file"
            )

        # Build messages array with conversation history
        messages = []

        # Add conversation history if exists
        if request.conversation_history:
            messages.extend(request.conversation_history)

        # Add current question
        messages.append({
            "role": "user",
            "content": request.question
        })

        # Call OpenAI API with GPT-4o-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        # Extract the assistant's response
        assistant_message = response.choices[0].message.content

        # Update conversation history
        updated_history = messages.copy()
        updated_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return ChatResponse(
            answer=assistant_message,
            conversation_history=updated_history
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
