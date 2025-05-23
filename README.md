# Interactive Persona-Based Recommender Simulation

This repo generates persona entries, simulates a user (LLM A) interacting with a recommender (LLM B), and verifies whether the model’s recommendation matches the ground truth.

## Project Structure

- **personas.py**  
  Generates a “benchmark entry” JSON for a given persona description.  
  Appends:
  - `products`: a list of ~30 items with attributes and a `"user_preference"` label  
  - `user_attributes`: decisive filters that rule out disliked items  
  - `correct_product` & `noise_traits` for evaluation (hidden from LLM B)

- **simulation_logic.py**  
  Core simulation functions:
  - `llm_a_respond(user_attributes, question)` → simulates user answers  
  - `llm_b_interact(products, user_attributes, model_name)` → runs 5 rounds of dynamic questioning and issues a recommendation

- **simulation.py**  
  Command-line driver:
  1. Takes `<persona_index>` (and optional `[llm_b_model]`, default `gpt-4o`).  
  2. Auto-generates `benchmark_entries/persona_<idx>.json` if missing.  
  3. Loads the persona file, runs `llm_b_interact`, and checks the recommendation against `correct_product`.

## Setup

1. **Clone** this repo and `cd` into it.  
2. **Install** dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install openai datasets python-dotenv