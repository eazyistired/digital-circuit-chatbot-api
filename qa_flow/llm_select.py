from transformers import AutoTokenizer, AutoModelForCausalLM

models_folder_path = "/mnt/Storage1/grozavu/digital-circuit-chatbot/models"


def get_tokenizer_and_model(model_name, quantization_config):
    tokenizer, model = None, None

    model_path = models_folder_path + "/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quantization_config, device_map={"": 0}
    )

    assert (
        tokenizer != None and model != None
    ), f"\n \n Tokenizer or LLM not loaded for model name: {model_name} \n \n"

    return tokenizer, model
