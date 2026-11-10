import json
from utils import call_llm

def _load_json_object(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(text[start:end + 1])

def extract_entities_and_relationship(text):
    system_prompt = f"""
        You are an expert agent in extracting entities and finding relationship between them.
    """
    user_prompt = f"""
        Analyze the text below. Extract important entities and their explicit semantic relationships.
        Sanitize entity names: remove quotes or escape characters.
        Return strictly a valid JSON object matching this exact structure:
        {{
            "entities": [
                {{"name": "Python", "type": "Language"}},
                {{"name": "PostgreSQL", "type": "Database"}}
            ],
            "relationships": [
                {{"source": "Python", "target": "PostgreSQL", "type": "CONNECTS_TO"}}
            ]
        }}
        Text content: {text}
        """

    try:
        response = call_llm(system_prompt, user_prompt, max_tokens=4096)
        return _load_json_object(response)
    except Exception as e:
        print(f"LLM Extraction failed: {e}")
        return {"entities": [], "relationships": []}




