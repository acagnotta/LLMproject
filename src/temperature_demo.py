"""Experiment: how temperature changes the output.

Asks the same creative question several times at different temperatures.
Run with:  python src/temperature_demo.py
"""
import ollama

MODEL = "llama3.2:3b"
PROMPT = "Invent a name for a barber shop run by robots. Reply with the name only."

for temperature in [0.0, 0.7, 1.5]:
    print(f"\n=== temperature = {temperature} ===")
    for _ in range(3):
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            options={"temperature": temperature},
        )
        print(" -", response["message"]["content"].strip())
