from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
from litegen import LLM

llm = LLM('dsollama')
app = FastAPI()

from weblair import google
from liteauto import parse

def google_search(query:str):
    """ perform websearch"""
    return parse(google(query,max_urls=1))[0].content



class TranscriptionRequest(BaseModel):
    text: str

@app.post("/process_text")
async def process_text(request: TranscriptionRequest):
    return {
        "response": llm(prompt = request.text,tools=[google_search]),
        "original_text": request.text
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)