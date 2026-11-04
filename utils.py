import os

def get_db_config():
    db_config = {
        "dbname": os.environ.get("DB_NAME"),
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
        "port": os.environ.get("DB_PORT"),
        "host": os.environ.get("DB_HOST"),
    }
    return db_config

def call_llm(system_prompt: str, user_prompt: str):
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL"),
                    api_key=os.environ.get("API_KEY"),
                    )

    response = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()