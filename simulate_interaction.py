from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_a_respond(user_attributes, question):
    """
    LLM A simulates the user based on user_attributes.
    """
    prompt = f"""You are simulating a user with the following attributes describing their preferences:
{json.dumps(user_attributes, indent=2)}

Answer the following question as this user would, in a short, natural language sentence:

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You simulate a user based on given attributes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def llm_b_interact(products, user_attributes, llm_b_model="gpt-4o"):
    """
    LLM B interacts with LLM A iteratively to find a liked product.
    """
    # Initialize chat messages for LLM B
    messages = [
        {"role": "system", "content": "You are a product recommender. Your goal is to ask the user questions to learn their preferences and recommend the best product from the provided list."},
        {"role": "assistant", "content": f"The available products are:\n{json.dumps(products, indent=2)}"}
    ]

    # Ask up to 5 clarifying questions
    for _ in range(5):
        q_resp = client.chat.completions.create(
            model=llm_b_model,
            messages=messages + [
                {"role": "assistant", "content": "Ask one concise question to clarify the user's preference based on the conversation so far."}
            ],
            temperature=0.7,
            max_tokens=100
        )
        question = q_resp.choices[0].message.content.strip()
        print(f"LLM B: {question}")

        answer = llm_a_respond(user_attributes, question)
        print(f"LLM A (user simulation): {answer}")

        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": answer})

    rec_resp = client.chat.completions.create(
        model=llm_b_model,
        messages=messages + [
            {"role": "assistant", "content": "Based on the conversation, recommend the single best product from the list. Provide the product details and a brief justification."}
        ],
        temperature=0.7,
        max_tokens=200
    )
    recommendation = rec_resp.choices[0].message.content.strip()
    return recommendation