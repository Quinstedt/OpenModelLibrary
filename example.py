import argparse
import os
from lib import Session, Model, NegativeTokenCountError, InsufficientAllowedTokensError, TokenError
from dotenv import load_dotenv

load_dotenv()

#############################################
# Set up arguments
#############################################
parser = argparse.ArgumentParser(description='Model parameters.')

parser.add_argument("-m", "--model", dest="model", type=str, default="mistral_debugger",
                    help="Set the model to use")
parser.add_argument('-t', '--temperature', dest="temperature", type=float, default=0.7,
                    help='Set the temperature value')

args = parser.parse_args()
model: str = args.model.lower()
temperature = args.temperature  # Default 0.7

#############################################
# Load model
#############################################

# Define a default path
default_model = "/mimer/NOBACKUP/groups/naiss2024-22-453/.cache/models--mistralai--Mistral-Small-Instruct-2409/snapshots/a5b29a334a0991ea7bb2d78ef9c7ca14ac9f8f04" 

if model == "mistral":
    model_path = os.getenv("MISTRAL")
elif model == "llama":
    model_path = os.getenv("LLAMA")
else: 
    model_path = default_model


model: Session = Session.create(
    name="Example Session",
    model_name_or_path=model_path,
    max_new_tokens=2048,  # required
    temperature=temperature
)


#############################################
# Prompt the model
#############################################

user_prompt = "Tell me a joke"
response = model.prompt(user_prompt) 
print("Model response", response)
