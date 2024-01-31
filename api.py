"""
# api.py usage

` python api.py -dr "123.wav" -dt "一二三。" -dl "zh" `

## 执行参数:

`-s` - `SoVITS模型路径, 可在 config.py 中指定`
`-g` - `GPT模型路径, 可在 config.py 中指定`

调用请求缺少参考音频时使用
`-dr` - `默认参考音频路径`
`-dt` - `默认参考音频文本`
`-dl` - `默认参考音频语种, "中文","英文","日文","zh","en","ja"`

`-d` - `推理设备, "cuda","cpu","mps"`
`-a` - `绑定地址, 默认"127.0.0.1"`
`-p` - `绑定端口, 默认9880, 可在 config.py 中指定`
`-fp` - `覆盖 config.py 使用全精度`
`-hp` - `覆盖 config.py 使用半精度`

`-hb` - `cnhubert路径`
`-b` - `bert路径`

## 调用:

### 推理

endpoint: `/`

使用执行参数指定的参考音频:
GET:
    `http://127.0.0.1:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`
POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

手动指定当次推理所使用的参考音频:
GET:
    `http://127.0.0.1:9880?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh&text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh",
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400


### 更换默认参考音频

endpoint: `/change_refer`

key与推理端一样

GET:
    `http://127.0.0.1:9880/change_refer?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh"
}
```

RESP:
成功: json, http code 200
失败: json, 400


### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
    `http://127.0.0.1:9880/control?command=restart`
POST:
```json
{
    "command": "restart"
}
```

RESP: 无

"""


import argparse
import os
import signal
import sys
from time import time as ttime
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Header, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

import uvicorn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from feature_extractor import cnhubert
from io import BytesIO
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
import config as global_config

import asyncio
import json
import shutil
import uuid
from datetime import timedelta
from typing import Optional
from typing import List
from pydantic import BaseModel
from aioredis import Redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import time
import aiofiles
import logging

# 定义一个锁，用于同步对 keys_count 的访问
keys_count_lock = asyncio.Lock()

log_file_path = 'api.log'

# 配置日志系统，将日志输出到文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=log_file_path,  # 日志文件路径
    filemode='a'  # 文件打开模式，'a' 代表追加，'w' 代表覆写
)

# 创建日志记录器
logger = logging.getLogger(__name__)

# 测试日志输出，信息将被写入到文件中
logger.info("信息将被记录到日志文件中。")


pool = None
status_key = "status"
results_key = "results"
expiration = 24 * 60 * 60  # Expire after 24 hours
keys_count = {}
keys_count_backup = {}
auth_keys = []

tmp_dir = "/tmp/audio_files"

class Task(BaseModel):
    model: str
    content: str
    timestamp: Optional[float]

g_config = global_config.Config()

with open('config.json', 'r') as f:
    local_config = json.load(f)
model_list =  local_config['model_list']

with open('auth_keys.txt', 'r') as f:
    auth_keys = [line.strip() for line in f.readlines()]


# AVAILABLE_COMPUTE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")

parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")

parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu / mps")
parser.add_argument("-a", "--bind_addr", type=str,
                    default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False

parser.add_argument("-hb", "--hubert_path", type=str,
                    default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str,
                    default=g_config.bert_path, help="覆盖config.bert_path")

args = parser.parse_args()

sovits_path = args.sovits_path
gpt_path = args.gpt_path

class DefaultRefer:
    def __init__(self, path, text, language):
        self.path = args.default_refer_path
        self.text = args.default_refer_text
        self.language = args.default_refer_language

    def is_ready(self) -> bool:
        return is_full(self.path, self.text, self.language)


default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)

device = args.device
port = args.port
host = args.bind_addr

if sovits_path == "":
    sovits_path = g_config.pretrained_sovits_path
    print(f"[WARN] 未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
    gpt_path = g_config.pretrained_gpt_path
    print(f"[WARN] 未指定GPT模型路径, fallback后当前值: {gpt_path}")

# 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
if default_refer.path == "" or default_refer.text == "" or default_refer.language == "":
    default_refer.path, default_refer.text, default_refer.language = "", "", ""
    print("[INFO] 未指定默认参考音频")
else:
    print(f"[INFO] 默认参考音频路径: {default_refer.path}")
    print(f"[INFO] 默认参考音频文本: {default_refer.text}")
    print(f"[INFO] 默认参考音频语种: {default_refer.language}")

is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half  # 炒饭fallback

print(f"[INFO] 半精: {is_half}")

cnhubert_base_path = args.hubert_path
bert_path = args.bert_path

cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def is_empty(*items):  # 任意一项不为空返回False
    for item in items:
        if item is not None and item != "":
            return False
    return True


def is_full(*items):  # 任意一项为空返回False
    for item in items:
        if item is None or item == "":
            return False
    return True


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # 输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T

def load_tts_model(gpt_path, sovits_path, device):
  n_semantic = 1024
  dict_s2 = torch.load(sovits_path, map_location="cpu")
  hps = dict_s2["config"]

  class DictToAttrRecursive:
    def __init__(self, input_dict):
      for key, value in input_dict.items():
        if isinstance(value, dict):
          # 如果值是字典，递归调用构造函数
          setattr(self, key, DictToAttrRecursive(value))
        else:
          setattr(self, key, value)

  hps = DictToAttrRecursive(hps)
  hps.model.semantic_frame_rate = "25hz"
  dict_s1 = torch.load(gpt_path, map_location="cpu")
  config = dict_s1["config"]
  ssl_model = cnhubert.get_model()
  if is_half:
    ssl_model = ssl_model.half().to(device)
  else:
    ssl_model = ssl_model.to(device)

  vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
  if is_half:
    vq_model = vq_model.half().to(device)
  else:
    vq_model = vq_model.to(device)
  vq_model.eval()
  print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
  hz = 50
  max_sec = config['data']['max_sec']
  t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
  t2s_model.load_state_dict(dict_s1["weight"])
  if is_half:
    t2s_model = t2s_model.half()
  t2s_model = t2s_model.to(device)
  t2s_model.eval()
  total = sum([param.nelement() for param in t2s_model.parameters()])
  print("Number of parameter: %.2fM" % (total / 1e6))
  return hps, ssl_model, vq_model, t2s_model, config, hz, max_sec

load_tts_model(gpt_path, sovits_path, device)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec


dict_language = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja"
}


def get_tts_wav(gpt_path, sovits_path, ref_wav_path, prompt_text, prompt_language, text, text_language):
  hps, ssl_model, vq_model, t2s_model, config, hz, max_sec = load_tts_model(gpt_path, sovits_path, device)

  t0 = ttime()
  prompt_text = prompt_text.strip("\n")
  prompt_language, text = prompt_language, text.strip("\n")
  zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half == True else np.float32)
  with torch.no_grad():
      wav16k, sr = librosa.load(ref_wav_path, sr=16000)
      wav16k = torch.from_numpy(wav16k)
      zero_wav_torch = torch.from_numpy(zero_wav)
      if (is_half == True):
          wav16k = wav16k.half().to(device)
          zero_wav_torch = zero_wav_torch.half().to(device)
      else:
          wav16k = wav16k.to(device)
          zero_wav_torch = zero_wav_torch.to(device)
      wav16k = torch.cat([wav16k, zero_wav_torch])
      ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
      codes = vq_model.extract_latent(ssl_content)
      prompt_semantic = codes[0, 0]
  t1 = ttime()
  prompt_language = dict_language[prompt_language]
  text_language = dict_language[text_language]
  phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
  phones1 = cleaned_text_to_sequence(phones1)
  texts = text.split("\n")
  audio_opt = []
  for text in texts:
    phones2, word2ph2, norm_text2 = clean_text(text, text_language)
    phones2 = cleaned_text_to_sequence(phones2)
    if (prompt_language == "zh"):
      bert1 = get_bert_feature(norm_text1, word2ph1).to(device)
    else:
      bert1 = torch.zeros((1024, len(phones1)), dtype=torch.float16 if is_half == True else torch.float32).to(
        device)
    if (text_language == "zh"):
      bert2 = get_bert_feature(norm_text2, word2ph2).to(device)
    else:
      bert2 = torch.zeros((1024, len(phones2))).to(bert1)
    bert = torch.cat([bert1, bert2], 1)

    all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
    bert = bert.to(device).unsqueeze(0)
    all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
    prompt = prompt_semantic.unsqueeze(0).to(device)
    t2 = ttime()
    with torch.no_grad():
      # pred_semantic = t2s_model.model.infer(
      pred_semantic, idx = t2s_model.model.infer_panel(
        all_phoneme_ids,
        all_phoneme_len,
        prompt,
        bert,
        # prompt_phone_len=ph_offset,
        top_k=config['inference']['top_k'],
        early_stop_num=hz * max_sec)
    t3 = ttime()
    # print(pred_semantic.shape,idx)
    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
    refer = get_spepc(hps, ref_wav_path)  # .to(device)
    if (is_half == True):
      refer = refer.half().to(device)
    else:
      refer = refer.to(device)
    # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
    audio = \
      vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0),
                       refer).detach().cpu().numpy()[
        0, 0]  ###试试重建不带上prompt部分
    audio_opt.append(audio)
    audio_opt.append(zero_wav)
  t4 = ttime()
  print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
  yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

async def handle(sovits_path, gpt_path, refer_wav_path, prompt_text, prompt_language, text, text_language):
    if (
            refer_wav_path == "" or refer_wav_path is None
            or prompt_text == "" or prompt_text is None
            or prompt_language == "" or prompt_language is None
    ):
        refer_wav_path, prompt_text, prompt_language = (
            default_refer.path,
            default_refer.text,
            default_refer.language,
        )
        if not default_refer.is_ready():
            return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)

    with torch.no_grad():
        gen = get_tts_wav(
            gpt_path, sovits_path, refer_wav_path, prompt_text, prompt_language, text, text_language
        )
        sampling_rate, audio_data = next(gen)

    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)

    torch.cuda.empty_cache()
    # torch.mps.empty_cache()
    if device == "mps":
        print('executed torch.mps.empty_cache()')
        torch.mps.empty_cache()
    return StreamingResponse(wav, media_type="audio/wav")


app = FastAPI()

def valid_auth_key(auth_key: str = Header(...)):  # Use depends to validate and count auth_key
    if not auth_key.startswith('Bearer '):
        raise HTTPException(status_code=400, detail='Invalid token schema')
    
    key = auth_key.split(' ')[1]
    if key not in auth_keys:
        raise HTTPException(status_code=401, detail='Invalid key')

    return key

@app.get("/model_list")
async def tts_endpoint(key: str = Depends(valid_auth_key)):
    model_list =  local_config['model_list']
    return model_list

@app.on_event("startup")
async def startup():
    global pool
    global keys_count_lock
    keys_count_lock = asyncio.Lock()
    try:
        pool = Redis.from_url("redis://localhost")
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
    os.makedirs(tmp_dir, exist_ok=True)
    asyncio.create_task(worker())
    asyncio.create_task(cleanup())
    ##启动定时器，每10分钟更新一次key的调用次数
    sched = AsyncIOScheduler()
    sched.add_job(update_keys_usage, IntervalTrigger(minutes=10))
    sched.start()


@app.on_event("shutdown")
async def shutdown():
    await pool.close()

async def save_key_usage_to_redis():
    async with keys_count_lock:
        global keys_count_backup

        # 保存每个key的计数到 Redis，并在本地备份再重置内存中的计数
        for key, count in keys_count.items():
            current_count = await pool.get(key) or 0
            await pool.set(key, int(current_count) + count)
        
        # 更新备份数据并清空原计数容器
        keys_count_backup = keys_count.copy()
        keys_count.clear()

async def update_keys_usage():
    async with keys_count_lock:
        global keys_count_backup

        # 如果备份为空，意味着自上次更新以来没有新的计数
        if not keys_count_backup:
            logger.warning("No keys to update found. 'keys_count_backup' is empty.")
            return

        # 从备份获取键的当前计数
        keys_to_update = {}
        for key in keys_count_backup:
            current_count = await pool.get(key) or 0
            keys_to_update[key] = int(current_count)
        
        if keys_to_update:
            logger.info(f"Number of keys to update: {len(keys_to_update)}.")
            with open('keys_count.txt', 'w') as f:
                json.dump(keys_to_update, f)
            logger.info("keys_count.txt file updated with Redis key usage.")
        else:
            logger.warning("No keys were found to update in 'keys_count_backup'.")
        
        # 在完成更新后清空备份
        keys_count_backup.clear()

async def handleTask(model: str, content: str):
    model_list =  local_config['model_list']
    if model not in model_list:
        model = 'default'

    model_root_path = local_config['model_root_path']
    prompt_text_path = f"{model_root_path}/{model}/{model}.txt"
    refer_wav_path = f"{model_root_path}/{model}/{model}.wav"
    sovits_path = f"{model_root_path}/{model}/{model}.pth"
    gpt_path = f"{model_root_path}/{model}/{model}.ckpt"
    print('content = ' + content)
    text = content

    with open(prompt_text_path, 'r',  encoding='utf-8') as file:
        prompt_text = file.read()
    prompt_language = 'zh'
    text_language = 'zh'

    response =  await handle(
        sovits_path,
        gpt_path,
        refer_wav_path,
        prompt_text,
        prompt_language,
        text,
        text_language,
    )
    if isinstance(response, StreamingResponse):
      audio_file_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.wav")
      async with aiofiles.open(audio_file_path, 'wb') as out_file:
          async for data in response.body_iterator:
              await out_file.write(data)
    
    return audio_file_path

@app.post("/task")
async def get_task_id(task: Task, key: str = Depends(valid_auth_key)):
    start_time = time.perf_counter()  # Start a timer to measure the request handling time
    logger.info("Task submission started.")
    if task.model not in model_list:
        raise HTTPException(status_code=400, detail='错误的模型名称')

    if len(task.content) > 100:
        raise HTTPException(status_code=400, detail='转换文本太长,超过100个字限制')

    task_id = str(uuid.uuid4())
    current_timestamp = time.time()

    await pool.hset(status_key, task_id, json.dumps({"status": "SUBMITTED", "timestamp": current_timestamp}))
    await pool.expire(status_key, expiration)
    await pool.rpush('queue', json.dumps({"task_id": task_id, 
                                          "content": task.content,
                                          "model": task.model,
                                          "timestamp": current_timestamp}))
    async with keys_count_lock:
        logger.info("Lock acquired.")
        keys_count[key] = keys_count.get(key, 0) + 1

    await save_key_usage_to_redis()

    end_time = time.perf_counter()  # Stop the timer after all operations are complete
    print(f"Task submission completed in {end_time - start_time:.4f} seconds.")
    return {"task_id": task_id, "status": "SUBMITTED"}

@app.get("/task/{task_id}")
async def status(task_id: str, key: str = Depends(valid_auth_key)):
    task_status = await pool.hget(status_key, task_id)
    if task_status is not None:
        task_status = json.loads(task_status)
        if task_status['status'] == "SUCCESS":
            audio_file_path = await pool.hget(results_key, task_id)
            if audio_file_path:
                audio_file_path = audio_file_path.decode()
                return StreamingResponse(file_generator(audio_file_path), media_type="audio/wav")
    return {
        "task_id": task_id,
        "status": task_status['status'] if task_status is not None else "UNKNOWN",
    }

async def file_generator(filename):
    async with aiofiles.open(filename, mode='rb') as file:
        while True:
            chunk = await file.read(4096)  # Or any other chunk size
            if not chunk:
                break
            yield chunk

async def worker():
    while True:
        task = await pool.lpop('queue')
        if task:
            task = json.loads(task)
            task_id = task['task_id']
            
            # 将耗时操作放到后台任务中去执行，不阻塞worker循环
            asyncio.create_task(process_task(task_id, task['model'], task['content']))

        # 适当的间隔或者使用 asyncio.sleep 来避免忙等
        await asyncio.sleep(1)

async def process_task(task_id: str, model: str, content: str):
    try:
        logger.info(f"Processing task {task_id}")
        await pool.hset(status_key, task_id, json.dumps({"status": "PROCESSING", "timestamp": time.time()}))
        await pool.expire(status_key, expiration)

        # 执行耗时的任务
        audio_file_path = await handleTask(model, content)

        # 任务完成后更新状态和结果
        await pool.hset(results_key, task_id, audio_file_path)
        await pool.hset(status_key, task_id, json.dumps({"status": "SUCCESS", "timestamp": time.time()}))
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        await pool.hset(status_key, task_id, json.dumps({"status": "FAILED", "timestamp": time.time()}))

    # 最后要设置过期
    await pool.expire(status_key, expiration)


# Cleanup task to remove expired audio files
async def cleanup():
    while True:
        now = time.time()
        for file in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, file)
            if os.path.getmtime(file_path) < now - expiration:
                os.remove(file_path)
        await asyncio.sleep(expiration)  # Run the cleanup task every 24 hours


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, workers=1)
