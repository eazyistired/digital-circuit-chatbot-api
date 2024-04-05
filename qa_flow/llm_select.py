from transformers import LlamaTokenizer, LlamaForCausalLM

models_folder_path = '/mnt/Storage1/grozavu/digital-circuit-chatbot/models'

def get_tokenizer_and_model(model_name, quantization_config):
    tokenizer, model = None, None

    match model_name:
        case 'llama-2-7b-hf':
            model_path = models_folder_path + '/' + model_name
            tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map={"": 0}
            )
        case 'llama-2-13b-hf':
            tokenizer = None
            model = None
        case _:
            print(f'ERROR \n model {model_name} not found')

    assert tokenizer != None and model != None
    
    return tokenizer, model