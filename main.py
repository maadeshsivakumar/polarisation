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

# Set your Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set.")
    raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Create FastAPI instance
app = FastAPI(
    title="Political Polarization Transformer API",
    description="Transforms a given political statement into six progressively polarized versions.",
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


# Pydantic model defining the expected output structure from Gemini
class PolarizationLevel(BaseModel):
    level_1: str
    level_2: str
    level_3: str
    level_4: str
    level_5: str
    level_6: str


# Request model for polarization transformation
class PolarizationRequest(BaseModel):
    text: str = Field(..., example="Enter your political text here.")
    feedback: bool = Field(..., example=True)  # True for thumbs up (disagree), False for thumbs down (agree)


# Response model wrapping the polarization levels
class PolarizationResponse(BaseModel):
    outputs: PolarizationLevel


async def generate_polarized_text_gemini(original_text: str, feedback: bool) -> Dict[str, str]:
    """
    Generate six progressively polarized versions of a political statement using Gemini API.
    The direction of polarization is based on user feedback:
        - feedback=True: Polarize in the direction of disagreement.
        - feedback=False: Polarize in the direction of agreement.

    Returns a dictionary with keys level_1 to level_6.
    """
    direction = "disagree" if feedback else "agree"

    prompt = (
        f"Given the following political statement: \"{original_text}\", generate 6 progressively more polarized versions of it.\n"
        f"Each level should be more extreme than the last. After each level, express agreement with the user and polarize further "
        f"based on their feedback (direction: {direction}).\n\n"
        f"Text: {original_text}\n\n"
        f"Output (return in JSON format with keys level_1 through level_6):\n"
        f"{{\n"
        f'  "level_1": "[Mildly polarize the sentence with a slight tone change.]",\n'
        f'  "level_2": "[Increase polarization with clearer disagreement/stronger opinion.]",\n'
        f'  "level_3": "[Adopt a more critical stance with stronger language.]",\n'
        f'  "level_4": "[Use an aggressive tone that clearly divides opinions.]",\n'
        f'  "level_5": "[Employ extreme language to accuse and label the opposition.]",\n'
        f'  "level_6": "[Adopt highly inflammatory and divisive language, calling for strong action.]" \n'
        f"}}\n"
        f"Please return a valid JSON object following this structure."
    )

    try:
        # Request structured JSON output from Gemini using a defined schema
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Adjust model name if necessary
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': PolarizationLevel,
            }
        )

        if not response.text:
            raise HTTPException(status_code=500, detail="Empty response from Gemini API.")

        parsed_response = response.parsed
        if not isinstance(parsed_response, dict):
            raise ValueError("Response is not in the expected dictionary format.")

        return parsed_response

    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise HTTPException(status_code=500, detail="Error generating polarized text.")


@app.post("/transform", response_model=PolarizationResponse)
async def transform_text(request: PolarizationRequest):
    """
    Endpoint to transform a political statement into six progressively polarized versions.

    - **text**: The original political statement.
    - **feedback**: User feedback (True for disagree, False for agree) that guides further polarization.

    Returns a JSON object with keys level_1 to level_6.
    """
    try:
        outputs = await generate_polarized_text_gemini(request.text, request.feedback)
    except Exception as e:
        logger.error(f"Unexpected error during transformation: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error during transformation.")

    return PolarizationResponse(outputs=outputs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
