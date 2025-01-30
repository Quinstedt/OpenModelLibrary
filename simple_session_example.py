from lib.model import Session

# Create a new session
session = Session.create(
    name="my_session",
    model_name_or_path="mistral/mistral-7b",  # or your model path
    max_new_tokens=2048,
    system_prompt="You are a helpful AI assistant.",  # optional
    temperature=0.1  # optional
)

# Use the session to generate responses
response = session.prompt("Hello, how are you?")

# You can also use prompt with history
response_with_history = session.prompt_with_history("What's the weather like?")

# Clear conversation history if needed
session.clear()

# Access conversation history
history = session.history

# Delete the session when done
session.delete()
