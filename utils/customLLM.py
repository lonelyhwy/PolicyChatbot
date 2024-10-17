from langchain.llms.base import LLM
from transformers import TextIteratorStreamer
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM

class CustomHFStreamingLLM(LLM):
    def __init__(self, model_name, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          device_map='auto', 
                                                          torch_dtype='auto'
                                                          )
        self.model_name = model_name  # Store model name for identifying params

    def _call(self, prompt, stop=None, run_manager=None):
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Run generation in a separate thread to avoid blocking
        generation_thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        generation_thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            if run_manager:
                run_manager.on_llm_new_token(new_text)
            else:
                # For debugging purposes
                print(new_text, end='', flush=True)

        generation_thread.join()
        return generated_text

    @property
    def _identifying_params(self):
        return {'model_name': self.model_name}

    @property
    def _llm_type(self):
        return 'custom_hf'