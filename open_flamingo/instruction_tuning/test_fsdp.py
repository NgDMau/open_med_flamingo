from transformers import AutoModelForCausalLM, MptConfig, MptModel
from transformers.models.mpt.modeling_mpt import MptForCausalLM
from torch.distributed.fsdp import fully_shard, FSDPModule
import torch.distributed as dist
import torch.distributed as dist
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

dist.init_process_group(backend="nccl", rank=0, world_size=1)


model = AutoModelForCausalLM.from_pretrained("anas-awadalla/mpt-7b", dtype="auto", device_map="auto")
print(type(model))
assert isinstance(model, MptForCausalLM)


for layer in model.transformer.blocks:
    fully_shard(layer)

fully_shard(model)

# Only check for FSDPModule after sharding
assert isinstance(model, FSDPModule)
print(model)

# Clean up distributed resources
dist.destroy_process_group()