from transformers import AutoTokenizer, AutoModelForCausalLM


def get_tokenizer_and_model(model_path, quantization_config):
    tokenizer, model = None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quantization_config, device_map={"": 0}
    )

    assert (
        tokenizer != None and model != None
    ), f"\n \n Tokenizer or LLM not loaded for model name: {model_path} \n \n"

    return tokenizer, model
