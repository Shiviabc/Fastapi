import ollama
import logging
from ai.base import AIPlatform # Make sure this import is correct for your project structure
from fastapi.concurrency import run_in_threadpool # Import the key function

class Ollama(AIPlatform):
    def __init__(self, model: str = "mistral"):
        self.model = model
        try:
            self.client = ollama.Client(host='http://127.0.0.1:11434')
            # It's good practice to confirm the connection here
            self.client.list()
            logging.info(f"OllamaPlatform initialized and connected with model: {self.model}")
        except Exception as e:
            # This makes startup failures more obvious
            logging.error(f"CRITICAL: Could not connect to Ollama during initialization: {e}")
            raise ConnectionError(f"Could not connect to Ollama during initialization: {e}") from e

    async def chat(self, prompt: str) -> str:
        try:
            logging.info(f"Sending prompt to Ollama model: {self.model}")

            # This is the synchronous function we need to run in a thread
            def sync_generate():
                response_data = self.client.generate(model=self.model, prompt=prompt, stream=False)
                logging.info("Received response from Ollama.")
                return response_data.get("response", "Error: No 'response' key in Ollama's output.")

            # Use run_in_threadpool to prevent blocking the server
            response = await run_in_threadpool(sync_generate)
            return response

        except Exception as e:
            logging.error(f"Error communicating with Ollama server: {e}")
            return f"Error: Could not connect to the Ollama server. Please ensure it is running."