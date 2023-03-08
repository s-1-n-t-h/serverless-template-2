from transformers import pipeline
import torch
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = ORTModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    #model = pipeline('summarization', model="google/pegasus-cnn_dailymail", framework='pt',
                     #device=device, use_auth_token='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')
    model = pipeline("summarization", model=model, tokenizer=tokenizer)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = model(prompt)

    # Return the results as a dictionary
    return result
