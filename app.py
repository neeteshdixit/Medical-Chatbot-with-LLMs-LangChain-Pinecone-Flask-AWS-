import os
import requests
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file (if you want to use an API key)
# load_dotenv()

app = Flask(__name__)
# Enable CORS for the frontend running on the local file system
CORS(app)

# --- Configuration ---
# NOTE: The Canvas environment automatically provides the API key to the fetch call.
# Leaving this blank is standard practice for the environment.
API_KEY = os.environ.get("GEMINI_API_KEY", "") 
# Define the base URL for the Gemini API
BASE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
# URL for the API call including the API key
API_URL_WITH_KEY = f"{BASE_API_URL}?key={API_KEY}"

# --- Gemini API Helper Function ---

def call_gemini_api(prompt: str, system_instruction: str, is_structured: bool = False, schema: dict = None):
    """
    Common function to call the Gemini API with structured or unstructured output.
    Implements exponential backoff for resilience.
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}], # Always use grounding for health questions
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {}
    }

    if is_structured and schema:
        # Configuration for receiving structured JSON output
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseSchema"] = schema

    max_retries = 5
    delay = 1

    for i in range(max_retries):
        try:
            response = requests.post(
                API_URL_WITH_KEY,
                headers={'Content-Type': 'application/json'},
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            # Extract the text (which might be JSON string if structured is True)
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'AI response failed to generate text.')
            
            if is_structured:
                # Attempt to parse the JSON string
                return json.loads(text)
            else:
                return text

        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and i < max_retries - 1:
                # print(f"Retrying in {delay}s due to error: {e}") # Do not log retries
                time.sleep(delay)
                delay *= 2
            else:
                raise e
        except requests.exceptions.RequestException as e:
            # Connection errors, timeouts
            return f"Error connecting to the Gemini API: {e}"
        except json.JSONDecodeError as e:
            # Error in parsing structured response
            return f"Error: Failed to parse AI response as JSON. Raw response: {text}"

    return "Error: Maximum number of retries reached for API request."

# --- Helper to format chat history ---
def format_history(history):
    """Converts the list of chat objects into a clean string format for the model."""
    return "\n".join([f"{item['role'].capitalize()}: {item['text']}" for item in history])

# --- Flask Routes ---

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles the standard conversational chat (Feature 1)."""
    data = request.get_json()
    user_query = data.get('query', '')
    history = data.get('history', [])

    if not user_query:
        return jsonify({"response": "Please provide a query."}), 400

    # Include the entire history in the prompt for conversational context
    full_prompt = f"Previous Conversation Context:\n{format_history(history)}\n\nNew User Query: {user_query}"
    
    # System instruction for general chat
    system_instruction = (
        "You are MediBot, a secure and professional AI health assistant. Your role is to provide general, "
        "helpful, and well-grounded health information based on the user's query and the conversation context. "
        "STRICT RULE: NEVER provide a definitive diagnosis, specific treatment plan, or replace a real doctor. "
        "Always start the response with the disclaimer: 'Disclaimer: I am an AI assistant and not a medical professional. "
        "Always consult a qualified doctor for diagnosis and treatment.' Answer clearly and concisely."
    )
    
    try:
        ai_response = call_gemini_api(full_prompt, system_instruction, is_structured=False)
        return jsonify({"response": ai_response})
    
    except Exception as e:
        print(f"An unexpected error occurred in chat route: {e}")
        return jsonify({"response": "An internal server error occurred while processing the AI request."}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_chat():
    """Summarizes the entire chat history (Feature 2)."""
    data = request.get_json()
    history = data.get('history', [])

    if len(history) < 2:
        return jsonify({"response": "Not enough conversation history to summarize."}), 400

    history_text = format_history(history)
    
    # System instruction for summarization
    system_instruction = (
        "You are MediBot's administrative assistant. Your task is to provide a brief, professional summary "
        "of the user's health concerns and the advice given by the bot so far. The summary must be a single, "
        "easy-to-read paragraph. Start the summary with 'Conversation Summary:'."
    )
    
    try:
        ai_response = call_gemini_api(history_text, system_instruction, is_structured=False)
        return jsonify({"response": ai_response})
    
    except Exception as e:
        print(f"An unexpected error occurred in summarize route: {e}")
        return jsonify({"response": "An internal server error occurred during summarization."}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_symptoms():
    """Analyzes symptoms from the chat history and returns a structured category and recommendation (Feature 3)."""
    data = request.get_json()
    history = data.get('history', [])

    if len(history) < 2:
        return jsonify({"response": "Not enough symptom data in history for analysis."}), 400

    history_text = format_history(history)
    
    # System instruction for structured symptom analysis
    system_instruction = (
        "You are a sophisticated medical triage AI. Analyze the user's symptoms described in the chat history. "
        "Categorize the situation into one of these three categories: 'Immediate Care (Emergency)', 'Monitor Closely (Moderate Risk)', "
        "or 'Common Ailment (Low Risk)'. Provide a brief reasoning for the category and a single, safe recommendation (e.g., 'Consult a doctor immediately' or 'Rest and hydration'). "
        "Your output MUST be a JSON object conforming to the provided schema."
    )

    # Define the required JSON structure (Schema)
    json_schema = {
        "type": "OBJECT",
        "properties": {
            "category": {"type": "STRING", "description": "The determined risk category: 'Immediate Care (Emergency)', 'Monitor Closely (Moderate Risk)', or 'Common Ailment (Low Risk)'."},
            "reasoning": {"type": "STRING", "description": "A brief justification (1-2 sentences) for the chosen category based on the symptoms."},
            "recommendation": {"type": "STRING", "description": "A single, safe, and actionable next step for the user."},
        },
        "required": ["category", "reasoning", "recommendation"]
    }
    
    try:
        json_response = call_gemini_api(history_text, system_instruction, is_structured=True, schema=json_schema)
        
        # Check if the response is a dict (parsed JSON) or an error string
        if isinstance(json_response, dict):
            return jsonify({"analysis": json_response})
        else:
            return jsonify({"response": f"Error: Could not process structured analysis. {json_response}"}), 500

    except Exception as e:
        print(f"An unexpected error occurred in analyze route: {e}")
        return jsonify({"response": "An internal server error occurred during symptom analysis."}), 500


@app.route('/', methods=['GET'])
def home_status():
    """Simple status check for the backend."""
    return jsonify({"status": "MediBot Backend is running and ready for API calls!"})

if __name__ == '__main__':
    # Setting use_reloader=False is often necessary to prevent dual launches in development environments
    print("Starting MediBot Backend on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
