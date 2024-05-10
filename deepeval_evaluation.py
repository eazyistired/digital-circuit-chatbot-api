from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from qa_flow.qr_handler import get_tokenizer_and_model
import transformers
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Initialize a test case
test_case = LLMTestCase(
    actual_output="""
Based on the datasheet provided, the inputs for the high side drivers in 6ED2230S12T are:

* HIN1, HIN2, and HIN3 (Logic input for high side gate driver output, in phase)

These inputs are located at pins 1, 2, and 3 of the IC, respectively.
                        
""",
    input="""
    How the faults are cleared in 6ED2230S12T?

""",
    retrieval_context=[
        """
                ['cause the IC to report a fault via the RF E pin. The first is an undervoltage condition of VCC  and the second is if the \nover -current feature has r ecognize d a fault.  Once the fault condition  occurs, the RF E pin is internally pulled to VSS  \nand the fault clear timer is activated. The RF E output stays in the low state until the fault condition has been removed \nand the fault clear timer expires; once the fault clear timer expires, the voltage on the RF E pin will return to its \nexternal pull -up voltage.   \nThe length of the fault clear time pe riod (t FLTCLR ) is determined by exponential charging characteristics of the \ncapacitor where the time constant is set by R RFE and C RFE. Figure 15  shows that R RFE is connected between the external \nsupply (V DD)1) and the RF E pin, while C RFE is placed between the RF E and VSS  pins . \n \n \n \n \n \n \n \n \n \n \n \n \n \nFigure 15  Programming the fault clear timer  \nVCC\nHIN (x3)\nRFE\nITRIP\nVSS COMLIN \nLO HO (x3)VB(x3)\n\n', 'cause the IC to report a fault via the RF E pin. The first is an undervoltage condition of VCC  and the second is if the \nover -current feature has r ecognize d a fault.  Once the fault condition  occurs, the RF E pin is internally pulled to VSS  \nand the fault clear timer is activated. The RF E output stays in the low state until the fault condition has been removed \nand the fault clear timer expires; once the fault clear timer expires, the voltage on the RF E pin will return to its \nexternal pull -up voltage.   \nThe length of the fault clear time pe riod (t FLTCLR ) is determined by exponential charging characteristics of the \ncapacitor where the time constant is set by R RFE and C RFE. Figure 15  shows that R RFE is connected between the external \nsupply (V DD)1) and the RF E pin, while C RFE is placed between the RF E and VSS  pins . \n \n \n \n \n \n \n \n \n \n \n \n \n \nFigure 15  Programming the fault clear timer  \nVCC\nHIN (x3)\nRFE\nITRIP\nVSS COMLIN \nLO HO (x3)VB(x3)\n\n', 'cause the IC to report a fault via the RF E pin. The first is an undervoltage condition of VCC  and the second is if the \nover -current feature has r ecognize d a fault.  Once the fault condition  occurs, the RF E pin is internally pulled to VSS  \nand the fault clear timer is activated. The RF E output stays in the low state until the fault condition has been removed \nand the fault clear timer expires; once the fault clear timer expires, the voltage on the RF E pin will return to its \nexternal pull -up voltage.   \nThe length of the fault clear time pe riod (t FLTCLR ) is determined by exponential charging characteristics of the \ncapacitor where the time constant is set by R RFE and C RFE. Figure 15  shows that R RFE is connected between the external \nsupply (V DD)1) and the RF E pin, while C RFE is placed between the RF E and VSS  pins . \n \n \n \n \n \n \n \n \n \n \n \n \n \nFigure 15  Programming the fault clear timer  \nVCC\nHIN (x3)\nRFE\nITRIP\nVSS COMLIN \nLO HO (x3)VB(x3)\n\n']
            """
    ],
)


class Llama13bHF(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda"  # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        # model.to(device)

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=100, do_sample=True
        )
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "llama-2-13b-hf"


tokenizer, model = get_tokenizer_and_model(
    model_path="/mnt/Storage1/grozavu/digital-circuit-chatbot/models/llama-2-13b-hf",
    quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True),
)

llama2_13b_hf = Llama13bHF(model=model, tokenizer=tokenizer)
print(llama2_13b_hf.generate("Write me a joke"))

metric = AnswerRelevancyMetric(model=llama2_13b_hf, threshold=0.5)
metric.measure(test_case)

print(metric.score)
print(metric.reason)
