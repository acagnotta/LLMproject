"""Interactive terminal chat with a local LLM, with conversation memory.

Run with:  python src/chat.py

Commands:
  /temp <value>   change temperature (e.g. /temp 1.2)
  /reset          forget the conversation
  /history        show how many messages / tokens are being sent
  /bye            quit
"""
import ollama

MODEL = "llama3.2:3b"
SYSTEM_PROMPT = "You are a helpful and concise assistant."


def new_history() -> list[dict]:
    """A conversation always starts with just the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def main() -> None:
    history = new_history()
    temperature = 0.7
    last_prompt_tokens = 0
    print(f"Chatting with {MODEL} (temperature={temperature}). Type /bye to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        # --- Commands -------------------------------------------------------
        if user_input == "/bye":
            break
        if user_input == "/reset":
            history = new_history()
            print("[conversation reset]\n")
            continue
        if user_input == "/history":
            print(f"[{len(history)} messages, last prompt = {last_prompt_tokens} tokens]\n")
            continue
        if user_input.startswith("/temp"):
            try:
                temperature = float(user_input.split()[1])
                print(f"[temperature set to {temperature}]\n")
            except (IndexError, ValueError):
                print("[usage: /temp 0.5]\n")
            continue

        # --- Normal message ---------------------------------------------------
        # 1. Append the user message to the history
        history.append({"role": "user", "content": user_input})

        # 2. Send the WHOLE history: the model has no memory of its own
        stream = ollama.chat(
            model=MODEL,
            messages=history,
            options={"temperature": temperature},
            stream=True,  # receive the answer token by token, like ChatGPT
        )

        # 3. Print tokens as they arrive and collect the full answer
        print("Assistant: ", end="", flush=True)
        answer = ""
        for chunk in stream:
            piece = chunk["message"]["content"]
            print(piece, end="", flush=True)
            answer += piece
            if chunk.get("done"):
                last_prompt_tokens = chunk.get("prompt_eval_count", 0)
        print("\n")

        # 4. Append the answer too, so the next turn "remembers" it
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
