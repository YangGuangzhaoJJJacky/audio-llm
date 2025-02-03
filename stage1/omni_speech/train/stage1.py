from omni_speech.model.builder import load_pretrained_model,create_model
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import whisper
from omni_speech.conversation import conv_templates
import ipdb  
import math
import json
from tqdm import tqdm
from omni_speech.datasets.preprocess import tokenizer_speech_token
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
from tqdm import tqdm
import torch.optim as optim
import torch.optim as optim
from transformers import DataCollatorForSeq2Seq
from torch.nn.utils.rnn import pad_sequence

# Custom dataset class

def collate_fn(batch):
    for i in range(len(batch)):
        batch[i]= batch[i].values()
        
    input_ids,labels,speech_tensors,speech_lengths = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=128009)
    labels = pad_sequence(labels, batch_first=True, padding_value=128009)

    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    return {"input_ids":input_ids,"labels":labels, "speech":speech_tensors, "speech_lengths":speech_lengths}

class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, model_config, input_type, mel_size):
        self.questions = questions
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size

    def __getitem__(self, index):
        item = self.questions[index]
        speech_file = item["speech"]
        qs = item["conversations"][0]["value"]
        re = item["conversations"][1]["value"]

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], re)
        prompt = conv.get_prompt()

        speech = whisper.load_audio(speech_file)
        if self.input_type == "raw":
            speech = torch.from_numpy(speech)
            if self.model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif self.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
        ret=dict(input_ids=input_ids,labels=input_ids, speech=speech.to(torch.bfloat16), speech_lengths=torch.LongTensor([speech.shape[0]]))
        return ret
    def __len__(self):
        return len(self.questions)
    
# DataLoader
def create_data_loader(questions, tokenizer, model_config, input_type, mel_size, batch_size=2, num_workers=1):
    # assert batch_size == 1, "batch_size must be 1"
    
    dataset = CustomDataset(questions, tokenizer, model_config, input_type, mel_size)
    #data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return dataset


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def train_model(args):
    # 设置每张卡的device
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'     # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps

    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, context_len = create_model(model_path, args.model_base, is_lora=args.is_lora, s2s=args.s2s)

    
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx) #chunk 1 chunk-idx 0 取list中的多少进行测试
    data_loader = create_data_loader(questions, tokenizer, model.config, args.input_type, args.mel_size)


    from transformers import Trainer, TrainingArguments
    # 初始化Trainer
    training_args = TrainingArguments(
    output_dir='saves',                         # 输出路径，包括模型检查点、中间文件等
        overwrite_output_dir=True,                  # 是否覆写 output_dir
        do_train=True,                              # 是否做训练
        do_eval=False,                               # 是否做评估
        eval_steps=1,                            # 评估步骤间隔
        per_device_train_batch_size=2,              # 每设备批次
        gradient_accumulation_steps=6,              # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快
        learning_rate=1e-4,
        weight_decay=0.01,
        adam_beta2=0.95,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine',                 # 学习率调度策略，LLM 训练一般都用余弦
        logging_steps=1,                           # 打印步骤间隔
        report_to=None,                             # 日志输出目标，不想用 wandb 可以设置为 None
        num_train_epochs=50,                         # 训练轮数，2 ~ 3 即可
        save_steps=1000,                            # 检查点保存步骤间隔
        save_total_limit=2,                         # output_dir 内留存的检查点最大数目
        seed=3407,                                   # 随机种子
        bf16=True                                  # 是否开启混合精度训练
        
    )
    tokenizer.pad_token = tokenizer.eos_token
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_loader,
        eval_dataset=data_loader,
        data_collator=collate_fn
    )
    trainer.train()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="raw")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--is_lora", action="store_true", default=False)
    args = parser.parse_args()
    train_model(args)