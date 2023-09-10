# reference codes https://huggingface.co/docs/transformers/tasks/asr
from datasets import load_dataset, Audio

minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
minds = minds.train_test_split(test_size=0.2)
minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])


from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

def uppercase(example):
    return {"transcription": example["transcription"].upper()}


minds = minds.map(uppercase)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch



import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")


import evaluate

wer = evaluate.load("wer")

import numpy as np


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}