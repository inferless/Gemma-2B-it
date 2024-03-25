from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from pathlib import Path
import os

class InferlessPythonModel:
    def initialize(self):
        repo_id = "google/gemma-2b-it" # Specify the model repository 
        HF_TOKEN = os.getenv("HF_TOKEN")  # Access Hugging Face token from environment variable
        
        model_dir = snapshot_download(
            repo_id,
            token=HF_TOKEN  # Provide token if necessary
        )

        # Define sampling parameters for model generation
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128,dtype="float16")

        # Initialize the LLM object
        self.llm = LLM(model=model_dir)
        
    def infer(self,inputs):
        prompts = inputs["prompt"]  # Extract the prompt from the input
        result = self.llm.generate(prompts, self.sampling_params)
        # Extract the generated text from the result
        result_output = [output.outputs[0].text for output in result]
        
        # Return a dictionary containing the result
        return {'generated_text': result_output[0]}

    def finalize(self):
        pass
