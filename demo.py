import torch


def run_llm():
    from vllm import LLM, SamplingParams
    llm = LLM(model="THUDM/chatglm3-6b", trust_remote_code=True, max_model_len=8192)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "希望这篇文章能",
        "给六岁小朋友解释一下万有引",
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    del llm


def run_transformer():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, torch_dtype=torch.float32).cuda()
    model = model.eval()
    tko = tokenizer("给六岁小朋友解释一下万有引")
    input_ids = torch.LongTensor([tko.input_ids]).to('cuda')

    outputs = model(input_ids, output_hidden_states=True)

    for i in range(len(outputs.hidden_states)):
        hidden_state = outputs.hidden_states[i]
        hidden_state = hidden_state.permute(1, 0, 2)
        print(f"gold {i}", hidden_state[..., :5])

    outputs = torch.argmax(outputs.logits, dim=-1).tolist()[0]
    print(outputs)
    response = tokenizer.decode(outputs)
    del model


if __name__ == '__main__':
    # check_rope()
    run_llm()
    run_transformer()
