import argparse
import os
import 

load_dotenv()

#############################################
# Set up arguments
#############################################
parser = argparse.ArgumentParser(description='Process some file paths.')
parser.add_argument(
    '-p','--paths',
    dest='paths', 
    type=str, 
    nargs='+', 
    help='at least one file path',
    required=True
)

parser.add_argument("-m", "--model", dest="model", type=str, default= "mistral_debugger",
                    help="Set the model to use")
parser.add_argument('-t', '--temperature', dest="temperature", type=str, default=0.7,
                    help='Path to the module file')

args = parser.parse_args()
model: str = args.model.lower()
temperature = args.temperature #Default 0.7

#############################################
# Load model
#############################################

from .util.model_lib import Session

# Define a default path
default_model = "" 

 max_new_tokens = 2048

if model == "mixtral":
        model_path = os.getenv("MIXTRAL")
elif model == "llama":
        model_path = os.getenv("LLAMA")
else: 
    model_path = default_model

system_prompt: str = """You are a helpful AI assistant"""

myModel: Model = Model.get(
    model_name_or_path = model,
    max_new_tokens,
    system_prompt,
    temperature
)
#############################################
# Prompt the model
#############################################

###########
# The library only provide the user prompt and response, if we want to also keep track of the used system prompt, 
# you will need to add it to the chatHistory
#
# chatHistory = [{"role": "system", "content": self.system_prompt}]
##############

prompt = "Tell me a joke"
chatHistory = []
response = model.get_response([], prompt)
print("Model response" , response)

