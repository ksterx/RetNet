import argparse
import os

from transformers import AutoTokenizer

from retnet.modeling_retnet import RetNetForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="/nas/y_sasaki/results/RetNet/retnet3b_ctxLen2048_datav1",
)
parser.add_argument("-c", "--checkpoint", type=str, default=None)
args = parser.parse_args()

path = os.path.join(args.model_path, args.checkpoint)

model = RetNetForCausalLM.from_pretrained(path)
model.push_to_hub(
    "Spiral-AI/RetNet-3b",
    revision=args.checkpoint,
    private=True,
)

tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")
tokenizer.push_to_hub("Spiral-AI/RetNet-3b", private=True)
