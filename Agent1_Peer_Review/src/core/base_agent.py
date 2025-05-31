from typing import Dict, Any, List
import json
import os
import requests # Import requests library
from datetime import datetime
#from openai import OpenAI # Removed OpenAI import
from dotenv import load_dotenv
from .config import DEFAULT_MODEL
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Determine the path to the .env file
# This is needed for the ValueError message if OPENROUTER_API_KEY is not found
from pathlib import Path
project_root = Path(__file__).resolve().parents[3] # Adjust if necessary
env_path = project_root / '.env'


class BaseReviewerAgent:
    """Base class for all reviewer agents."""
    
    def __init__(self, model=DEFAULT_MODEL):
        """
        Initialize the base reviewer agent.
        
        Args:
            model (str): The language model to use
        """
        self.name = self.__class__.__name__
        self.category = "Unknown"
        self.model = model
        
        # Fetch OpenRouter API key from environment
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError(f"OPENROUTER_API_KEY environment variable not set. Please check {env_path}")
        
        # Store OpenRouter API URL and headers
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "HTTP-Referer": "http://localhost:3000",  # Optional placeholder
            "X-Title": "Agent1 Peer Review"  # Optional placeholder
        }
        
    def llm(self, prompt: str) -> str:
        """Call the configured language model API (OpenRouter) with the given prompt."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert academic reviewer. Provide detailed analysis in JSON format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
            # Note: response_format={"type": "json_object"} is specific to OpenAI and removed.
            # OpenRouter models are generally expected to follow instructions for JSON output.
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload  # requests library handles json.dumps internally
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            response_json = response.json()

            # Verify the expected structure and extract content
            if not response_json.get("choices") or not response_json["choices"][0].get("message") or not response_json["choices"][0]["message"].get("content"):
                raise Exception("Invalid response structure from language model.")

            return response_json["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            # Handle network errors, timeouts, etc.
            raise Exception(f"Error calling language model API: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            # Handle errors in parsing response or unexpected structure
            raise Exception(f"Error processing response from language model: {str(e)}")
        except Exception as e:
            # Catch any other unexpected errors
            raise Exception(f"An unexpected error occurred in llm method: {str(e)}")
    
    def analyze_section(self, text: str, section_name: str) -> Dict[str, Any]:
        """Analyze a specific section of the manuscript.
        
        Args:
            text (str): Text content to analyze
            section_name (str): Name of the section being analyzed
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        prompt = f"""As a {self.name}, analyze the following {section_name} section:

{text}

Provide your analysis in the following JSON format:
{{
    "score": <1-5>,
    "remarks": [
        "List of specific issues, questions, or observations"
    ],
    "concrete_suggestions": [
        "List of actionable steps for improvement"
    ],
    "automated_improvements": [
        "List of AI-generated improvements"
    ]
}}

Ensure your response is valid JSON and includes all required fields."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"You are a {self.name} reviewer. Your response must be a JSON object string."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            response_json = response.json()
            
            # Verify the expected structure and extract content string
            if not response_json.get("choices") or \
               not response_json["choices"][0].get("message") or \
               not response_json["choices"][0]["message"].get("content"):
                raise ValueError("Invalid response structure from language model.")

            content_string = response_json["choices"][0]["message"]["content"]

            # Extract JSON from the content string (as the prompt requests JSON output)
            # This logic remains from the original method.
            start_idx = content_string.find('{')
            end_idx = content_string.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                analysis = json.loads(content_string[start_idx:end_idx])
            else:
                # If the LLM didn't return a string that contains a JSON object,
                # or if the string is not what we expect.
                raise ValueError("No valid JSON object found in the LLM's response content.")
            
            return analysis
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling language model API in analyze_section: {e}")
        except json.JSONDecodeError as e:
            # This can happen if response_json is not valid JSON, or content_string is not valid JSON
            print(f"Error decoding JSON in analyze_section: {e}. Response content: '{response_json if 'response_json' in locals() else 'N/A'}' ; LLM content_string: '{content_string if 'content_string' in locals() else 'N/A'}'")
        except (KeyError, IndexError, ValueError) as e:
            # Handles issues with expected keys in response_json or content_string parsing
            print(f"Error processing response structure in analyze_section: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred in analyze_section: {e}")

        return {"score": 0, "remarks": [], "concrete_suggestions": [], "automated_improvements": []}