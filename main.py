# main.py
import logging
import os
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your Gemini API key and initialize the Gemini client
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    logger.error("GEMINI_API_KEY environment variable not set.")
    raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=gemini_api_key)

# Create FastAPI app instance
app = FastAPI(
    title="Political Polarization Transformer API",
    description="API that transforms a given political text into 6 outputs with incremental degrees of polarization.",
    version="1.0.0"
)

# Allow CORS for local development (modify as needed in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the schema for the polarization levels using Pydantic
class PolarizationLevel(BaseModel):
    level_1: str
    level_2: str
    level_3: str
    level_4: str
    level_5: str
    level_6: str


# Request model
class PolarizationRequest(BaseModel):
    text: str = Field(..., example="Enter your political text here.")
    feedback: bool = Field(..., example=True)  # True for thumbs up (disagree), False for thumbs down (agree)


# Response model
class PolarizationResponse(BaseModel):
    outputs: PolarizationLevel


# Helper function to call Gemini API
async def generate_polarized_text_gemini(original_text: str, feedback: bool) -> Dict[str, str]:
    """
    Generate 6 progressively polarized paragraphs of the text in a single API call.
    The `feedback` determines if we polarize further in the direction of agreement or disagreement.
    """
    direction = "agree" if feedback else "disagree"

    prompt = (
        f"Given the following political statement: \"{original_text}\", generate 6 progressively more polarized "
        f"versions of it.\n"
        f"The polarization should increase at each level, with each level being more extreme than the last.\n"
        f"After each level, you should {direction} with the user's direction, and polarize the idea further according "
        f"to their feedback.\n"
        f"Start with the base statement and modify it based on the feedback. Continue polarizing the statement in the "
        f"direction of {direction}.\n\n"
        f"Text: {original_text}\n\n"
        f"Output: \n"
        f"{{\n"
        f"  \"level_1\": \"[Mildly polarize the idea of the sentence, keeping it neutral with slight tone change.]\",\n"
        f"  \"level_2\": \"[Increase the polarization, showing a clearer disagreement/stronger opinion.]\",\n"
        f"  \"level_3\": \"[A more critical stance with stronger language to highlight opposing views.]\",\n"
        f"  \"level_4\": \"[Aggressive tone, clearly dividing opinions with harsh language.]\",\n"
        f"  \"level_5\": \"[Use extreme language that accuses and labels the opposition.]\",\n"
        f"  \"level_6\": \"[Highly inflammatory and divisive language, calling for action or a strong response.]\"\n"
        f"}}\n"
        f"Please return the full response in a valid JSON format as shown above."
    )

    try:
        # Generate all levels in one call with the response schema defined
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Replace with the correct Gemini model name if needed
            contents=prompt,
            config={
                'response_mime_type': 'application/json',  # Specify that we want the output in JSON format
                'response_schema': PolarizationLevel,  # Specify the expected schema for the response
            }
        )

        # Check if response is valid and properly structured
        if not response.text:
            raise HTTPException(status_code=500, detail="Empty response from Gemini API.")

        # Return the parsed response directly as a structured JSON
        return response.parsed

    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise HTTPException(status_code=500, detail="Error generating polarized text.")


# Main endpoint to process the political text input
@app.post("/transform", response_model=PolarizationResponse)
async def transform_text(request: PolarizationRequest):
    """
    Accepts a political text input and returns six versions of the text, each with an increasing degree of polarization.
    The text is progressively polarized based on the user's feedback (agree or disagree).
    """
    original_text = request.text
    feedback = request.feedback  # True for thumbs up (disagree), False for thumbs down (agree)

    try:
        # Generate all six polarization levels in a single API call
        outputs = await generate_polarized_text_gemini(original_text, feedback)
    except HTTPException as http_err:
        # Re-raise HTTP exceptions
        raise http_err
    except Exception as e:
        logger.error(f"Unexpected error during transformation: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error during transformation.")

    return PolarizationResponse(outputs=outputs)


# For running with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
