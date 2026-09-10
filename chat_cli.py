import requests

API_URL = "http://127.0.0.1:5000/api/ai-advisor"


def main():
    print("AgriTech AI Chat")
    print("Type 'exit' to quit.\n")

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
                json={"query": question},
                timeout=180
            )

            data = response.json()

            if response.status_code == 200:
                print("\nAI:")
                print(data.get("ai_response", "No response received."))
                print()
            else:
                print("\nError:")
                print(data.get("error", "Something went wrong."))
                print()

        except requests.exceptions.ConnectionError:
            print(
                "\nCould not connect to the Flask backend.\n"
                "Make sure ai_assistant.py is running.\n"
            )

        except requests.exceptions.Timeout:
            print("\nAI response took too long.\n")

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()