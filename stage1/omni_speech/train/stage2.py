import argparse
import os
# import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0，1号GPU）
import torch
from torch.utils.data import Dataset, DataLoader
import whisper
# import ipdb  
import math
import json
from tqdm import tqdm
from omni_speech.conversation import conv_templates
from omni_speech.model.builder import load_pretrained_model,create_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
from tqdm import tqdm
import torch.optim as optim
# from memory_profiler import profile
import torch.optim as optim
from transformers import DataCollatorForSeq2Seq
import os
from torch.nn.utils.rnn import pad_sequence

# Custom dataset class

def collate_fn(batch_data):
    for i in range(len(batch_data)):
        batch_data[i] = batch_data[i].values()
    input_ids,labels,speech_tensors, tgt_units,speech_lengths = zip(*batch_data)

    # input_idspad为llama的<|eot_id|>
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=128009)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    tgt_units = pad_sequence(tgt_units, batch_first=True, padding_value=-100)
    # input_ids = torch.stack(input_ids, dim=0)
    # labels = torch.stack(labels, dim=0)
    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    #转fp16
    
    ret=dict(input_ids=input_ids,labels=labels, speech=speech_tensors.bfloat16(), tgt_units = tgt_units, speech_lengths=speech_lengths)
    return ret

class CustomDataset(Dataset):
    def __init__(self, questions, responses, tokenizer, model_config, input_type, mel_size):
        self.questions = questions
        self.responses = responses
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size

    
    # def get_tgt_unit(self, file_path):
    #     unique_data_list = [] 
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         for line in file:
    #             line = line.strip()
    #             parts = line.split('<')
    #             result = [part for part in parts if part and '>' in part]
    #             # 移除元素末尾的 '>'
    #             result = [part.split('>')[0] for part in result]
    #             line_list = [int(item) for item in result]                
    #             #unique_data = [line_list[i] for i in range(len(line_list)) if i == 0 or line_list[i] != line_list[i-1]]
    #             unique_data_list.append(line_list)
    #     # return torch.tensor(unique_data_list)
    #     return unique_data_list

    def __getitem__(self, index):
        #tgt_unit = torch.tensor(self.tgt_unit[index])
        responses = self.responses[index]
        prediction = responses['prediction']
        tgt_unit = responses['prediction_units']
        tgt_unit = torch.tensor([int(item) for item in tgt_unit.split(' ')])
        item = self.questions[index]
        speech_file = item["speech"]
        qs = item["conversations"][0]["value"]
        ans = item["conversations"][1]["value"]
        # llm_gt = self.llm_gt[index]
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], prediction)
        prompt = conv.get_prompt()
        

        speech = whisper.load_audio(speech_file)
        if self.input_type == "raw":
            speech = torch.from_numpy(speech)
            if self.model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif self.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        input_ids_ = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
        input_ids = input_ids_.tolist()
        # 处理 input_ids 和 labels，仅训练answer部分的loss 
        split_markers = [128006, 78191, 128007, 271]
        last_marker_index = -1

        for i in range(len(input_ids) - len(split_markers) + 1):
            if input_ids[i:i + len(split_markers)] == split_markers:
                last_marker_index = i + len(split_markers)
                break
        if last_marker_index != -1:
            list1 = input_ids[:last_marker_index]
            list2 = input_ids[last_marker_index:]

        labels = len(list1) * [-100] + list2
        labels = torch.tensor(labels, device=input_ids_.device, dtype=input_ids_.dtype)
        ret=dict(input_ids=input_ids_,labels=labels, speech=speech, tgt_units=tgt_unit ,speech_lengths=torch.LongTensor([speech.shape[0]]))
        # ret=dict(input_ids=input_ids,labels=None, speech=speech, tgt_units=tgt_unit ,speech_lengths=torch.LongTensor([speech.shape[0]]))
        return ret
    def __len__(self):
        return len(self.questions)
    
# DataLoader
def create_data_loader(questions, responses,tokenizer, model_config, input_type, mel_size, batch_size=1, num_workers=1):
    assert batch_size == 1, "batch_size must be 1"
    
    dataset = CustomDataset(questions,responses, tokenizer, model_config, input_type, mel_size)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'     # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps
    #local_rank = torch.distributed.get_rank()
    #torch.cuda.set_device(local_rank)
    #device = torch.device(f'cuda:{local_rank}')
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, context_len = create_model(model_path, args.model_base, device=device, is_lora=args.is_lora, s2s=args.s2s)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx) #chunk 1 chunk-idx 0 取list中的多少进行测试
    with open(os.path.expanduser(args.answer_file), "r") as f:
        responses = f.readlines()
        for i in range(len(responses)):
            responses[i] = json.loads(responses[i])
    data_loader = create_data_loader(questions,responses, tokenizer, model.config, args.input_type, args.mel_size)

    
    # optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # 学习率变大
    optimizer = optim.Adam(model.speech_generator.parameters() , lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)




    # 初始化Trainer
    model.train()
    training_args = TrainingArguments(
    output_dir='saves/stage2_fp16',                         # 输出路径，包括模型检查点、中间文件等
        overwrite_output_dir=True,                  # 是否覆写 output_dir
        do_train=True,                              # 是否做训练
        do_eval=False,                               # 是否做评估
        eval_steps=100,                            # 评估步骤间隔
        per_device_train_batch_size=1,              # 每设备批次
        gradient_accumulation_steps=8,              # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快
        learning_rate=3e-5,                         # 学习率大小
        lr_scheduler_type='cosine',                 # 学习率调度策略，LLM 训练一般都用余弦
        bf16=torch.cuda.is_bf16_supported(),        # 尝试配置 bf16
        fp16=not torch.cuda.is_bf16_supported(),    # bf16 不行就上 fp16
        half_precision_backend='cuda_amp',
        logging_steps=1,                           # 打印步骤间隔
        report_to=None,                             # 日志输出目标，不想用 wandb 可以设置为 None
        num_train_epochs=1000,                         # 训练轮数，2 ~ 3 即可
        save_steps=1000,                            # 检查点保存步骤间隔
        save_total_limit=100,                         # output_dir 内留存的检查点最大数目
        seed=3407,                                  # 随机种子
        max_grad_norm=1.0,

    )
    tokenizer.pad_token = tokenizer.eos_token
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_loader,
        eval_dataset=data_loader,
        data_collator=collate_fn,
        optimizers=(optimizer, None)
    )
    # with torch.no_grad:
    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="Llama-3.1-8B-Omni")
    parser.add_argument("--model-base", type=str, default='Llama-3.1-8B-Omni')
    # parser.add_argument("--question-file", type=str, default="./omni_speech/infer/examples/question.json")
    parser.add_argument("--question-file", type=str, default="data.json")
    parser.add_argument("--answer-file", type=str, default="omni_speech/infer/gen_answer_data/answer.json")
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=True)
    parser.add_argument("--is_lora",type=bool, default=False)
    #parser.add_argument("--local-rank")
    args = parser.parse_args()
    train_model(args)
