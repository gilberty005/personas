import sys
import os
import json
from datasets import load_dataset
from personas import generate_benchmark_entry
from simulate_interaction import llm_b_interact

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python simulation.py <persona_index> [llm_b_model]")
        sys.exit(1)

    idx = sys.argv[1]
    # Default model for LLM B
    llm_b_model = sys.argv[2] if len(sys.argv) == 3 else "gpt-4o"

    file_path = os.path.join("benchmark_entries", f"persona_{idx}.json")
    if not os.path.exists(file_path):
        print(f"Benchmark entry file {file_path} not found. Generating benchmark entry.")
        os.makedirs("benchmark_entries", exist_ok=True)
        dataset = load_dataset("Tianyi-Lab/Personas", split="train")
        persona = dataset[int(idx)]
        user_description = persona["Llama-3.1-70B-Instruct_descriptive_persona"]
        result = generate_benchmark_entry(user_description)
        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved generated benchmark entry to {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    user_attributes = data.get("user_attributes", [])
    products = data.get("products", [])

    if not user_attributes or not products:
        print("Invalid benchmark entry format: missing 'user_attributes' or 'products'.")
        sys.exit(1)

    recommendation = llm_b_interact(products, user_attributes, llm_b_model)
    print("\nLLM B: " + recommendation)

    correct = data.get("correct_product", {})
    correct_name = correct.get("name")
    if correct_name and correct_name in recommendation:
        print("Simulation result: Correct recommendation.")
    else:
        print(f"Simulation result: Incorrect. Expected recommendation: {correct_name}")

if __name__ == "__main__":
    main()
