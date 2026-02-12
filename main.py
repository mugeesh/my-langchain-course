import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)


def main():
    print("Hello from langchain-course!")
    print(f"Key: {os.environ.get('DEEPSEEK_API_KEY')}")


if __name__ == "__main__":
    main()
