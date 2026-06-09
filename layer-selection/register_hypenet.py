"""
register_hypenet.py — Register HypeNet models with Transformers AutoClasses.

Call this before using trust_remote_code with saved HypeNet models.
"""
from transformers import AutoConfig, AutoModel
from modeling.hypenet.configuration_hypenet import HypeNetConfig
from modeling.hypenet.modeling_hypenet import HypeNetForCausalLM

AutoConfig.register("hypenet", HypeNetConfig)
AutoModel.register(HypeNetConfig, HypeNetForCausalLM)

print("✅ HypeNet models registered with Transformers")
