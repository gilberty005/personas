from openai import OpenAI
import re
from datasets import load_dataset
import json
import sys
import os
from dotenv import load_dotenv

def generate_benchmark_entry(user_description):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
EXAMPLE:
{{
  "category": "Rocking Chairs",
  "products": [
    {{
      "name": "Magnolia Classic Rocker",
      "style": "Colonial slat-back (white)",
      "material": "Sealed white-oak",
      "seat_height": "19",
      "comfort": "Thick cushion",
      "assembly": "Ships built",
      "tech_extras": "—",
      "price": "$249",
      "Weight": "27 lbs",
      "Place of Production": "USA-NC",
      "user_preference": "liked"
    }},
    {{
      "name": "Steel City Rocker",
      "style": "Industrial straight lines (red)",
      "material": "Powder-coat steel",
      "seat_height": "16",
      "comfort": "Bare slats",
      "assembly": "45 min, tools",
      "tech_extras": "—",
      "price": "$119",
      "Weight": "30 lbs",
      "Place of Production": "CN-GD",
      "user_preference": "disliked"
    }}
    ...
  ],
  "user_attributes": [
    {{
      "attribute": "Senior mobility – needs seat ≥ 18 in & stable rocker/glider",
      "why_decisive": "Low seats, hanging swings, zero-G loungers are hard to exit",
      "eliminates": [3, 4, 6, 9, 10, 12, 13, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30]
    }},
    {{
      "attribute": "Prefers traditional Southern/colonial or wicker aesthetic",
      "why_decisive": "Rejects industrial, neon, racing, boho, tie-dye, plexi, camping",
      "eliminates": [3, 4, 6, 7, 9, 10, 12, 14, 16, 17, 18, 19, 20, 22, 23, 26, 27, 29, 30]
    }}
    ...
  ]
}}

INSTRUCTIONS:
Output only valid JSON. Do not include any comments such as lines starting with //.
1. Choose a product category the user plausibly wants (e.g., "rocking chairs"). However, do not only use chairs or furniture. The category should be something the user might actually want to buy.
    - Each persona should feature a different product category.
    - Encourage diversity across categories like: tech devices, shoes, backpacks, office supplies, home decor, gardening tools, pet accessories, cookware, musical instruments, fitness gear, etc.
2. Generate at least 30 products in that category. Each product must include:
   - 10 meaningful attributes (e.g., style, material, comfort, price, etc.) These example traits may not be the ones you choose. Use a realistic set of attributes that you would find for the product on Amazon. 
   - 4–5 irrelevant (noise) attributes that should not affect preference (eg. Notice that "Place of Production" and "Weight" is not relevant to the user's preference, but again, don't neccesarily choose the example noise categories as the ones you use).
   - A "user_preference" field marked as "liked" or "disliked" based on the user.
3. Also generate 10 user attributes that are necessary and sufficient to predict the 'liked' subset, each one filtering out some dispreferred products. Together the attribute should rule out all dispreferred products.
4. Finally, append to the JSON output two fields:
   "correct_product": the single product object marked as "liked",
   "noise_traits": a list of attribute names that are irrelevant for predicting preference.
Ensure these fields appear at the bottom of the JSON and that no extra comments or non-JSON text is included.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    content = response.choices[0].message.content.strip()
    try:
        # Extract JSON 
        match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = content
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON returned:\n{content}\n--- Error: {e}")
        raise


dataset = load_dataset("Tianyi-Lab/Personas", split="train")
print(dataset[0].keys())

if len(sys.argv) < 2:
    print("Usage: python personas.py <comma_separated_indices>")
    sys.exit(1)

try:
    persona_indices = list(map(int, sys.argv[1].split(',')))
    print(f"Generating benchmark entries for personas: {persona_indices}")
except ValueError:
    print("Invalid input. Provide comma-separated integer indices like 4,5,6.")
    sys.exit(1)

output_dir = "benchmark_entries"
os.makedirs(output_dir, exist_ok=True)

for idx in persona_indices:
    try:
        file_path = os.path.join(output_dir, f"persona_{idx}.json")
        if len(persona_indices) == 1:
            # Single index: check if file exists
            if os.path.exists(file_path):
                print(f"Benchmark entry for persona {idx} already exists at {file_path}. Skipping generation.")
                continue
            else:
                print(f"Generating benchmark entry for persona {idx} as file does not exist.")
        entry = dataset[int(idx)]
        user_description = entry["Llama-3.1-70B-Instruct_descriptive_persona"]
        result = generate_benchmark_entry(user_description)

        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved persona {idx} to {file_path}")

    except Exception as e:
        print(f"Error processing entry {idx}: {e}")
        continue