import torch
from retnet.modeling_retnet import RetNetForCausalLM
from transformers import AutoTokenizer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Spiral-AI/Spiral-RetNet-3b-base")
    model = RetNetForCausalLM.from_pretrained(
        "Spiral-AI/Spiral-RetNet-3b-base",
        device_map="auto",
    )

    while True:
        input_text = input("Input: ")

        if input_text == "":
            break
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=32,
                repetition_penalty=1.2,
            )
        print("Output:", tokenizer.decode(generated[0][len(input_ids[0]) :]))
