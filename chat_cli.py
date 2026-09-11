import requests

API_URL = "http://127.0.0.1:5000/api/ai-advisor"


def main():
    print("AgriTech AI Chat")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Chat ended.")
            break

        if not question:
            continue

        try:
            response = requests.post(
                API_URL,
                json={
                    "query": question,
                    "history": history
                },
                timeout=180
            )

            data = response.json()

            if response.status_code == 200:
                ai_reply = data.get(
                    "ai_response",
                    "No response received."
                )

                print("\nAI:")
                print(ai_reply)
                print()

                history.append({
                    "role": "user",
                    "content": question
                })

                history.append({
                    "role": "assistant",
                    "content": ai_reply
                })

                # Keep only recent conversation
                history = history[-10:]

            else:
                print("\nError:")
                print(
                    data.get(
                        "error",
                        "Something went wrong."
                    )
                )
                print()

        except requests.exceptions.ConnectionError:
            print(
                "\nCould not connect to Flask backend.\n"
                "Make sure ai_assistant.py is running.\n"
            )

        except requests.exceptions.Timeout:
            print("\nAI response took too long.\n")

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()