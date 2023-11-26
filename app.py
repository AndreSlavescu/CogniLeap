from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional
from cohere_chat import cohere_response_request
from image_understanding import ImageUnderstanding
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# load model so it doesn't need to load on first request
image_understanding = ImageUnderstanding("")  # pass in dummy path


@app.post("/process_question/")
async def process_question(user_question: str = Form(None)):
    assert isinstance(user_question, str), "user question must be str"

    # Ask question to cohere
    response = cohere_response_request(message=user_question)
    return {"query": user_question, "response": response}


@app.post("/process_image/")
async def process_image(image: UploadFile = File(...), user_question: str = Form(None)):
    assert isinstance(user_question, str), "user question must be str"

    os.makedirs('temp_images', exist_ok=True)

    # Save image
    file_location = f"temp_images/{image.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(image.file.read())

    # change path to appropriate image path
    image_understanding.path = file_location
    image_context = image_understanding.final_master_prompt(
        user_query=user_question
    )

    # Process the combined query using Cohere
    response = cohere_response_request(
        message=user_question,
        image_context=image_context
    )

    return {"query": user_question, "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
