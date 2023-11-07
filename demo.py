import torch


def run_llm(prompt):
    from vllm import LLM, SamplingParams
    llm = LLM(model="THUDM/chatglm3-6b", trust_remote_code=True, max_model_len=8192)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    outputs = llm.generate(prompt, sampling_params)

    # Print the outputs.
    for output in outputs:
        prom = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prom!r}, Generated text: {generated_text!r}")
    del llm


def run_transformer(prompt):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
    model = model.eval()
    tko = tokenizer(prompt)
    input_ids = torch.LongTensor([tko.input_ids]).to('cuda')

    outputs = model(input_ids, output_hidden_states=True)

    for i in range(len(outputs.hidden_states)):
        hidden_state = outputs.hidden_states[i]
        hidden_state = hidden_state.permute(1, 0, 2)
        print(f"gold {i}", hidden_state[..., 3, :5])
    del model


if __name__ == '__main__':
    prompt = "给六岁小朋友解释一下万有引"
    run_llm(prompt)
    print("-" * 100)
    run_transformer(prompt)
