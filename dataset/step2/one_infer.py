from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SimpleInference:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()
        
    def generate(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,      
            do_sample=True,
            temperature=0.7,
            seed=42  
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return response

if __name__ == "__main__":
    llm = SimpleInference("/home/recosele/Development/End2End/stage1/Llama-3-ELYZA-JP-8B")
    text_part ="会話調で、できるだけ短く1、2文で回答してください。あなたは健康に関する質問に答えることができるアシスタントです。1時間ほど前、左の肋骨の下あたりで筋肉のけいれんのような痛みを感じました。"
    audio_part = "その痛みは約1分間続いた後、消えました。それ以降、他の痛みや不快感はありません。これは軽度の心臓発作の症状だったのでしょうか？また、心臓発作の症状は一つだけ現れるものですか、それとも複数の症状が同時に現れるものですか？"
    input_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{text_part}<|eot_id|><|start_header_id|>user<|end_header_id|>{audio_part}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    response = llm.generate(input_text)
    print(response)