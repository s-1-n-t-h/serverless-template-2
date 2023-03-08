# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import pipeline
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM


def download_model():
    model = ORTModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    # do a dry run of loading the huggingface model, which will download weights
    #pipeline('summarization', model="google/pegasus-cnn_dailymail", framework='pt',use_auth_token='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')
    pipeline("summarization", model=model, tokenizer=tokenizer)

if __name__ == "__main__":
    download_model()
