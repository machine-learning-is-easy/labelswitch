# reference codes https://huggingface.co/docs/transformers/tasks/asr


from transformers import AutoModelForCTC

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)