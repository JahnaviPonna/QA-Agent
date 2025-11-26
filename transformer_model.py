# backend/transformer_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LocalHFModel:
    def __init__(self, model_name="google/flan-t5-base", device=None):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def generate(self, prompt: str, max_tokens=512):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_length=max_tokens,
            num_beams=4,
            temperature=0.0,
            early_stopping=True
        )

        raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        raw = raw.replace("\n", " ").strip()
        return raw
