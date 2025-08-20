import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class LegalChatbot(LLM):

    '''
    Google's Gemini Pro is a text-only LLM
    '''

    model: str = "models/gemini-2.0-flash-lite"
    temperature: float = 0.4
    max_output_tokens: int = 2000

    def __init__(self, **kwargs):
         super().__init__(**kwargs)
         self._configure_api()

    def _configure_api(self):
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=GEMINI_API_KEY)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            model = genai.GenerativeModel(
                self.model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens         
                }
            )
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return f"Error calling Gemini API: {str(e)}"
        
    @property
    def _llm_type(self) -> str:
        return "models/gemini-2.0-flash-lite"
    
if __name__ == "__main__":
        try:
            llm = LegalChatbot()
            response = llm("Explain tenancy laws in Maharastra in simple terms.")
            print("\nGemini response: ", response)
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
            print(f"Error: {str(e)}") 