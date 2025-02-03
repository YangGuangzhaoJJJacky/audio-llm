# temple="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\
# You are a helpful language and speech assistant.You are able to understand the speech content that the user provides,\
# and assist the user with a variety of tasks using natural language.<|eot_id|>\
# <|start_header_id|>user<|end_header_id|><speech>Please answer the questions in the user’s input speech.<|eot_id|>\
# <|start_header_id|>assistant<|end_header_id|><response><|end_of_text|>"



#读取目录下所有wav文件,并打印出来路径
import os
import json
wav_dir = 'wavs'
wav_files = os.listdir(wav_dir)

#打开文件并读取内容，和wav_files同时遍历
with open('r_100.txt', 'r', encoding='utf-8') as f:
    responses = f.readlines()
with open("data.json", "w", encoding="utf-8") as file:
    saved_array = []
    for wav,response in zip(wav_files,responses):
        #以json格式保存os.path.join(wav_dir,wav),response.strip()
        data={"id":wav.split('.')[0],
                "speech":os.path.join(wav_dir,wav),
                "conversations":[
                    {
                        "from": "human",
                        "value": "<speech>\nPlease directly answer the questions in the user's speech."
                    },
                    {   "from": "assistant",
                        "value": response.strip()
                    }
                ]
                }
        saved_array.append(data)
    json.dump(saved_array, file, indent=4, ensure_ascii=False)
    
    
    
    
    
    



            
