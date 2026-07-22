import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field, ValidationError

# ==========================================
# CHECKPOINT 1: API Integration & ENV Vars
# ==========================================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("XƏTA: HF_TOKEN .env faylında tapılmadı! Lütfən .env faylını yoxlayın.")

# Hugging Face Serverless Client (Pulsuz)
client = InferenceClient(api_key=hf_token)
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"


# ==========================================
# CHECKPOINT 5: Schema Definition for JSON Parsing
# ==========================================
class SupportResponseSchema(BaseModel):
    category: str = Field(description="Issue category: Tech, Billing, or General")
    sentiment: str = Field(description="Customer sentiment: Positive, Neutral, Negative")
    response_draft: str = Field(description="Polite customer support response draft")
    action_items: list[str] = Field(description="Action steps for the agent")


# ==========================================
# MAIN APP FUNCTION
# ==========================================
def generate_support_response(customer_query: str):
    print("\n--- 1. PROMPT ENGINEERING & REQUEST ---")
    
    system_prompt = """You are an AI Customer Support Assistant.
Analyze the customer's query and provide a structured JSON response.

Strict Rules:
1. Return ONLY valid JSON matching this exact structure:
{
  "category": "Tech" | "Billing" | "General",
  "sentiment": "Positive" | "Neutral" | "Negative",
  "response_draft": "string",
  "action_items": ["string"]
}
2. Do NOT add markdown, backticks (```json), or text outside the JSON.

Few-Shot Example:
User: "I was charged twice for my subscription this month!"
JSON Output:
{
  "category": "Billing",
  "sentiment": "Negative",
  "response_draft": "Dear customer, we apologize for the double charge. We are initiating a refund immediately.",
  "action_items": ["Verify double payment in portal", "Process refund"]
}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Customer Query: \"{customer_query}\""}
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )
    except Exception as e:
        print(f"API Call Failed: {e}")
        return

    # CHECKPOINT 6: Basic Cost & Token Usage Tracking
    usage = completion.usage
    prompt_tokens = usage.prompt_tokens if usage else 120
    completion_tokens = usage.completion_tokens if usage else 80
    total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)
    
    cost = 0.00  # Hugging Face Serverless API is free

    print(f"[TOKEN USAGE] Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total_tokens}")
    print(f"[ESTIMATED COST] ${cost:.6f} (Free Open Source Tier)")

    raw_response = completion.choices[0].message.content.strip()

    # CHECKPOINT 5: Output Parsing & Validation
    print("\n--- 2. OUTPUT PARSING & VALIDATION ---")
    try:
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(clean_json)
        validated_data = SupportResponseSchema(**parsed_json)
        print(" Valid JSON Schema successfully validated!")
        print(json.dumps(validated_data.model_dump(), indent=2))
    except (json.JSONDecodeError, ValidationError) as err:
        print(f" Parsing/Validation Error: {err}")
        print(f"Raw Output: {raw_response}")


# ==========================================
# CHECKPOINT 3: Streaming Response Management
# ==========================================
def stream_support_response(customer_query: str):
    print("\n--- 3. STREAMING RESPONSE DEMO ---")
    messages = [
        {"role": "system", "content": "You are a helpful customer support agent. Answer concisely."},
        {"role": "user", "content": customer_query}
    ]

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            stream=True
        )
        print("Streaming response: ", end="")
        for chunk in stream:
            # FIX: Boş paket gəldikdə xəta verməməsi üçün yoxlanış
            if len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content is not None:
                    print(content, end="", flush=True)
        print("\n\n Stream completed successfully.")
    except Exception as e:
        print(f"\nStreaming failed: {e}")


# RUN THE CODE
if __name__ == "__main__":
    sample_query = "My internet disconnects every 10 minutes, and I need this fixed for my work meeting!"
    generate_support_response(sample_query)
    stream_support_response(sample_query)