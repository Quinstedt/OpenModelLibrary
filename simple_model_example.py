from lib.model import Model

# Create a new model instance
model = Model.get(
    model_name_or_path="mistral/mistral-7b",  # or your model path
    max_new_tokens=2048,  # required
    context_window=4096,  # optional, defaults to 4096
    temperature=0.1  # optional, defaults to 0.1
)

# Prepare conversation history 
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Your prompt here"}
]

# Generate response
response = model.get_response(
    history=messages,
    prompt="Your prompt",
    dynamic_max_tokens=False  # optional parameter
)
