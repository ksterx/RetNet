if __name__ == "__main__":
    import argparse

    from gstop import GenerationStopper
    from transformers import AutoTokenizer

    from retnet.modeling_retnet import RetNetForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", type=str, default="")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="/nas/k_ishikawa/results/RetNet/retnet3b_ctxLen2048_anonym/checkpoint-280",
    )
    parser.add_argument("-max", "--max-tokens", type=int, default=128)
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model = RetNetForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")

    stopper = GenerationStopper(
        stop_tokens={
            "USER: ": [26168, 27, 208],
            " USER: ": [2975, 1203, 27, 208],
            "ASSISTANT: ": [14719, 9078, 18482, 27, 208],
            " ASSISTANT: ": [345, 4670, 9078, 18482, 27, 208],
            "\n": [186],
        },
        tokenizer_name=tokenizer,
    )

    def generate(user_input: str, history: str) -> str:
        prompt_input = f"USER: {user_input}\nASSISTANT: "
        prompt = history + prompt_input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.custom_generate(**inputs, max_new_tokens=128)
        response = tokenizer.decode(outputs[0])
        return response[len(prompt) :]

    # def chat(system=""):
    #     history = ""
    #     while True:
    #         user_input = input("USER: ")
    #         if user_input == "exit":
    #             break
    #         elif user_input == "clear":
    #             history = ""
    #             print("=" * 80)
    #             continue
    #         if system:
    #             history = system + "\n\n" + history
    #         response = generate(user_input, history)
    #         print(f"ASSISTANT: {stopper.format(response)}")
    #         history += f"USER: {user_input}\nASSISTANT: {response}\n"
    #         if system:
    #             print(history)
    #             history = history[len(system) + 2 :]

    # chat(system=args.system)

    def chat(system=""):
        history = ""
        while True:
            user_input = input("USER: ")
            if user_input == "exit":
                break
            elif user_input == "clear":
                history = ""
                print("=" * 80)
                continue
            if system:
                history = system + history
            response = generate(user_input, history)
            print(f"ASSISTANT: {stopper.format(response)}")
            history += f"USER: {user_input}\nASSISTANT: {response}\n"
            if system:
                history = history[len(system) :]

    chat(
        system="""\
ASSISTANTの名前は「みきお」です。
USERの名前は「ミクリ」です。

"""
    )
