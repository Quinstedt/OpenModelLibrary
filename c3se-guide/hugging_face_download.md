
## Download from hugging-face

Log in to hugging-face 

``` <apptainer path or alias> huggingface-cli ```

Generate a token in hugging-face and use it login. Make sure to be in the llm-readability folder before executing the command.

runenv huggingface-cli donwload --cache-dir ../cache /<model_name>

*The model name can be found by pressing the copy button next to the models repo name in hugging face. 
