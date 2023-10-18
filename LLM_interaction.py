import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerationConfig
from peft import PeftModel
import datasets
from typing import List

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops:List = []):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):

        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

class chat_model():
    def __init__(self, modelName:str, cache_dir:str=None):
        self.tokenizer, self.model = self.load_model(modelName, cache_dir)

    def load_model(self, modelName, cache_dir):
        tokenizer = AutoTokenizer.from_pretrained(modelName, use_fast=False, trust_remote_code=True, cache_dir = cache_dir, local_files_only = True)
        model = AutoModelForCausalLM.from_pretrained(modelName, local_files_only = True, device_map="auto", trust_remote_code=True, cache_dir = cache_dir).eval()
        model.generation_config = GenerationConfig.from_pretrained(modelName, cache_dir = cache_dir)
        model.generation_config.do_sample = True
        model.generation_config.repetition_penalty = 1.05
        model.generation_config.temperature = 0.7

        return tokenizer, model
    
    def lord_Lora(self, loraPath:str):
        self.model = PeftModel.from_pretrained(self.model, loraPath)

    def chat(self, text:str, stop_str:str):
        input_len = len(text)
        input_ids = self.tokenizer(text, return_tensors='pt')
        input_ids = input_ids.to("cuda:0")
        
        stop_ids = self.tokenizer.encode(stop_str)
        Stopping_Criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = torch.tensor([stop_ids]).to("cuda:0"))])
        gen = self.model.generate(**input_ids, stopping_criteria=Stopping_Criteria)

        gen_text = self.tokenizer.decode(gen.cpu()[0], skip_special_tokens=True)[input_len:]
        return gen_text
    
def main():
    chater = chat_model("F:/huggingface_model/ccyh123/Qwen-7B-Chat")
    respond = chater.chat("0 0 0 1 1 1: 2, 1 1 1 2 2 2: 3, 2 2 2 3 3 3: ", ", ")
    print(respond)


if __name__ == "__main__":
    main()
