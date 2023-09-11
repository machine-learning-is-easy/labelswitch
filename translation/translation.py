# code reference https://huggingface.co/docs/transformers/tasks/translation
from transformers import AdamWeightDecay
from transformers import AutoModelForSeq2SeqLM, GPT2Model
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model = AutoModelForSeq2SeqLM()

