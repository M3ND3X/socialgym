import google.generativeai as genai
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
import time
import random
import logging

# Rate limits per model (requests per minute)
MODEL_RPM_LIMITS = {
    'gemma': 30,
    'gemini-1.5-flash-8b': 2000,
    'gemini-1.5-flash': 2000,
    'gemini-1.5-pro': 1000,
    'gemini-2.0-flash-lite': 30000,
    'gemini-2.0-flash': 30000,
}

class ModelInterface:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_output_tokens: int,
        family: str,
        **kwargs
    ):
        """Initialize model interface"""
        load_dotenv()
        
        # Load all available API keys
        self.api_keys = self._load_api_keys()
        if not self.api_keys:
            raise ValueError("No API keys found in environment variables")
            
        # Store current API key index
        self.current_key_index = 0
        
        # Configure initial API
        self._configure_api()
        
        # Format model name correctly
        if not model_name.startswith(("models/", "tunedModels/")):
            model_name = f"models/{model_name}"
        
        # Store model parameters
        self.model_name = model_name
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        
        # Initialize model and get model info
        self.model = genai.GenerativeModel(model_name)
        self.model_info = genai.get_model(model_name)
        
        # Error handling parameters
        self.max_retries = 2
        self.retry_delay = 5  # Reduced from 30 seconds to 5 seconds
        
        # Token tracking
        self.total_tokens_used = 0
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        
    def _load_api_keys(self) -> List[str]:
        """Load all available API keys from environment"""
        keys = []
        i = 1
        while True:
            key = os.getenv(f"GOOGLE_API_KEY{i}")
            if not key:
                break
            keys.append(key)
            i += 1
        return keys
        
    def _configure_api(self):
        """Configure API with current key"""
        genai.configure(api_key=self.api_keys[self.current_key_index])
        
    def _rotate_api_key(self):
        """Rotate to next available API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_api()
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return self.model.count_tokens(text).total_tokens
        except Exception as e:
            logging.warning(f"Failed to count tokens: {e}")
            return 0
            
    def _check_token_limits(self, prompt: str) -> bool:
        """Check if prompt exceeds token limits"""
        prompt_tokens = self._count_tokens(prompt)
        if prompt_tokens > self.model_info.input_token_limit:
            logging.warning(
                f"Prompt exceeds input token limit: {prompt_tokens} > {self.model_info.input_token_limit}"
            )
            return False
        return True
        
    def _update_token_counts(self, response) -> None:
        """Update token usage statistics"""
        if hasattr(response, 'usage_metadata'):
            self.prompt_tokens_used += response.usage_metadata.prompt_token_count
            self.completion_tokens_used += response.usage_metadata.candidates_token_count
            self.total_tokens_used += response.usage_metadata.total_token_count
            
    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage statistics"""
        return {
            "total_tokens": self.total_tokens_used,
            "prompt_tokens": self.prompt_tokens_used,
            "completion_tokens": self.completion_tokens_used
        }
        
    def generate_response(
        self, 
        system_prompt: str, 
        user_prompt: str,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Generate response from the model with error handling and retries"""
        try:
            # Combine prompts
            prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Check token limits
            if not self._check_token_limits(prompt):
                raise ValueError("Prompt exceeds token limits")
            
            # Generate response with error handling
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle network errors
                if any(err in error_msg for err in ["unavailable", "socket", "connection", "timeout"]):
                    if retry_count < self.max_retries:
                        retry_count += 1
                        delay = min(2 ** retry_count + random.uniform(0, 1), 60)  # Exponential backoff with jitter
                        print(f"\nNetwork error. Retrying in {delay:.1f} seconds... ({retry_count}/{self.max_retries})")
                        time.sleep(delay)
                        return self.generate_response(system_prompt, user_prompt, retry_count)
                    else:
                        raise Exception(f"Network error after {self.max_retries} retries: {e}")
                
                # Re-raise other errors
                raise e
            
            # Update token counts
            self._update_token_counts(response)
            
            # Proactively throttle to respect RPM limits across all API keys
            # Determine plain model name
            model_name_plain = self.model_name.split('/')[-1]
            # Find RPM limit for this model
            rpm_limit = None
            for prefix, limit in MODEL_RPM_LIMITS.items():
                if model_name_plain.startswith(prefix):
                    rpm_limit = limit
                    break
            # If a limit is specified, sleep accordingly
            if rpm_limit and self.api_keys:
                delay = 60.0 / (rpm_limit * len(self.api_keys))
                time.sleep(delay)
            
            # Get token usage for this request
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', None),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', None),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', None)
            }
            
            # Extract and return relevant information
            return {
                "raw_response": response.text,
                "token_usage": token_usage,
                "total_tokens_used": self.total_tokens_used
            }
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            raise
        except Exception as e:
            error_msg = str(e)
            
            # Handle quota exceeded error
            if "429" in error_msg:
                if self.current_key_index < len(self.api_keys) - 1:
                    print(f"API quota exceeded. Rotating to next API key...")
                    self._rotate_api_key()
                    # Call generate_response with same retry count
                    return self.generate_response(system_prompt, user_prompt, retry_count)
                    
                elif retry_count < self.max_retries:
                    retry_count += 1
                    print(f"All API keys exhausted. Waiting {self.retry_delay} seconds before retry {retry_count}/{self.max_retries}...")
                    time.sleep(self.retry_delay)
                    # Call generate_response with incremented retry count
                    return self.generate_response(system_prompt, user_prompt, retry_count)
                    
            # Handle internal server error
            elif "500" in error_msg and retry_count < self.max_retries:
                retry_count += 1
                delay = random.randint(5, 15)  # Random delay between 5-15 seconds
                print(f"Internal server error. Retrying in {delay} seconds... ({retry_count}/{self.max_retries})")
                time.sleep(delay)
                # Call generate_response with incremented retry count
                return self.generate_response(system_prompt, user_prompt, retry_count)
                
            # If all retries exhausted or other error, raise the exception
            raise e

    def _retry_generate_response(self, system_prompt: str, user_prompt: str, retry_count: int) -> Dict[str, Any]:
        """Helper method to retry generate_response without recursion"""
        return self.generate_response(system_prompt, user_prompt, retry_count) 