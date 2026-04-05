import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=200,
    messages=[
        {"role": "user", "content": "Reply with exactly: API connection working."}
    ]
)

print(message.content[0].text)