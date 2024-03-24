from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from pathlib import Path
import os

class InferlessPythonModel:
    def initialize(self):
        repo_id = "meta-llama/Llama-2-13b-chat-hf"  # Specify the model repository ID
        HF_TOKEN = os.getenv("HF_TOKEN")  # Access Hugging Face token from environment variable
        VOLUME_NFS = os.getenv("VOLUME_NFS")  # Define model storage location
        model_dir = f"{VOLUME_NFS}/{repo_id}"  # Construct model directory path
        model_dir_path = Path(model_dir)  # Convert path to Path object

        # Create the model directory if it doesn't exist
        if not model_dir_path.exists():
            model_dir_path.mkdir(exist_ok=True, parents=True)

        # Download the model snapshot from Hugging Face Hub
        snapshot_download(
            repo_id,
            local_dir=model_dir,
            token=HF_TOKEN  # Provide token if necessary
        )

        # Define sampling parameters for model generation
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

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
