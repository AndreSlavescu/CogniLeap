import os
from openai import OpenAI

# OpenAI key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# GPT-4 turbo call
def gpt_4_turbo_answer(
    messages,
    model="gpt-4-1106-preview",
    max_tokens=750,
    temperature=0.6,
    top_p=0.9,
    frequency_penalty=1.2,
    presence_penalty=0.5,
):
    """
    GPT-4 turbo call
    """
    completion_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
    }

    response = client.chat.completions.create(**completion_params)

    return response.choices[0].message.content
