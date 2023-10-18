import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"            # 禁用显卡

import torch
print(torch.cuda.is_available())
from modelscope import snapshot_download             # 国内用户下载

model_checkpoint = "qwen/Qwen-7B"                    # 进入modelscope官网选择模型
cache_dir = "./"
model_path = snapshot_download(model_checkpoint, cache_dir = cache_dir, revision='v1.0.0')  # revision 需要与线上模型一致

print(f"这是该模型的路径：{model_path}, 将它作为--model_path传给agent.py")
print(f"This is the path of the download model: {model_path}. Use it as the --model_path parameter to agent.py")
