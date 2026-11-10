"""Step 2: first call to an LLM (local and free, via Ollama).

Prerequisites:
  1. Ollama server running (Ollama app open, or `ollama serve`)
  2. Model downloaded:  ollama pull llama3.2:3b

Run with:  python src/hello_llm.py
"""
import ollama

MODEL = "llama3.2:3b"   # 3B parameters, ~2 GB quantized: runs fine on a laptop


def ask(question: str, system: str = "You are a concise assistant.", temperature: float = 0.0) -> str:
    """Send a single question to the model and return its answer."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},   # global behavior instructions
            {"role": "user", "content": question},
        ],
        options={
            "temperature": temperature,  # 0 = more deterministic, 1 = more creative
            "num_predict": 500,          # max number of tokens in the ANSWER
        },
    )
    print(f"[tokens in: {response['prompt_eval_count']}, out: {response['eval_count']}]")
    return response["message"]["content"]


if __name__ == "__main__":
    print(ask("What is an embedding? Answer in 3 sentences."))

    # Experiment: same question, different system prompt
    print(ask("What is an embedding?", system="Explain it as if talking to an 8-year-old."))
