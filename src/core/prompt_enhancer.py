#!/usr/bin/env python3
"""
Prompt Enhancer Module - Uses GPT-2 to enhance prompts for diffusion models
"""

import logging
import torch
from typing import Optional, List
import os

logger = logging.getLogger(__name__)

class PromptEnhancer:
    """Uses GPT-2 to enhance prompts for better diffusion model results."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize GPT-2 model and tokenizer."""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            logger.info(f"🚀 Loading GPT-2 model: {self.model_name}")
            logger.info(f"💻 Device: {self.device}")
            
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ GPT-2 model loaded successfully!")
            
        except ImportError as e:
            logger.warning(f"⚠️ Transformers not available: {e}")
            logger.info("📦 Install with: pip install transformers")
            self.model = None
        except Exception as e:
            logger.warning(f"❌ Failed to load GPT-2 model: {e}")
            self.model = None
    
    def enhance_prompt(self, original_prompt: str, max_length: int = 100, 
                      enhancement_type: str = "diffusion", max_tokens: int = 77) -> str:
        """
        Enhance a prompt using GPT-2 for better diffusion model results.
        
        Args:
            original_prompt: The original prompt to enhance
            max_length: Maximum length of generated text
            enhancement_type: Type of enhancement ("diffusion", "cartoon", "detailed")
            max_tokens: Maximum tokens for the final prompt (default: 77 for diffusion models)
        
        Returns:
            Enhanced prompt string limited to max_tokens
        """
        if not self.model or not self.tokenizer:
            logger.warning("⚠️ GPT-2 model not available, returning original prompt")
            return original_prompt
        
        try:
            # Create enhancement prompt based on type
            if enhancement_type == "diffusion":
                enhancement_prompt = f"Rewrite this prompt for a diffusion model to improve image quality: {original_prompt}"
            elif enhancement_type == "cartoon":
                enhancement_prompt = f"Rewrite this prompt for a cartoon-style diffusion model to improve image quality: {original_prompt}"
            elif enhancement_type == "detailed":
                enhancement_prompt = f"Add more visual details to this prompt for better image generation: {original_prompt}"
            else:
                enhancement_prompt = f"Rewrite this prompt for a diffusion model to improve image quality: {original_prompt}"
            
            # Encode input prompt
            inputs = self.tokenizer.encode(enhancement_prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate enhanced text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated tokens
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the enhanced part (remove the original prompt)
            enhanced_part = generated_text.replace(enhancement_prompt, "").strip()
            
            # If enhancement failed or is too short, return original
            if len(enhanced_part) < 10:
                logger.warning("⚠️ GPT-2 enhancement too short, using original prompt")
                return original_prompt
            
            # Combine original with enhancement
            final_prompt = f"{original_prompt}, {enhanced_part}"
            
            # Limit to max_tokens for diffusion model compatibility
            final_prompt = self._limit_tokens(final_prompt, max_tokens)
            
            logger.info(f"🎯 Original prompt: {original_prompt}")
            logger.info(f"🎯 Enhanced prompt: {final_prompt}")
            logger.info(f"🎯 Token count: {len(self.tokenizer.encode(final_prompt))}")
            
            return final_prompt
            
        except Exception as e:
            logger.error(f"❌ Error enhancing prompt: {e}")
            return original_prompt
    
    def enhance_multiple_prompts(self, prompts: List[str], 
                               enhancement_type: str = "diffusion") -> List[str]:
        """
        Enhance multiple prompts using GPT-2.
        
        Args:
            prompts: List of original prompts
            enhancement_type: Type of enhancement
            
        Returns:
            List of enhanced prompts
        """
        enhanced_prompts = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"🎯 Enhancing prompt {i+1}/{len(prompts)}")
            enhanced = self.enhance_prompt(prompt, enhancement_type=enhancement_type)
            enhanced_prompts.append(enhanced)
        
        return enhanced_prompts
    
    def is_available(self) -> bool:
        """Check if GPT-2 model is available."""
        return self.model is not None and self.tokenizer is not None
    
    def _limit_tokens(self, prompt: str, max_tokens: int) -> str:
        """
        Limit prompt to maximum token count for diffusion model compatibility.
        
        Args:
            prompt: The prompt to limit
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Prompt truncated to max_tokens
        """
        if not self.tokenizer:
            return prompt
        
        # Encode the prompt to get tokens
        tokens = self.tokenizer.encode(prompt)
        
        # If within limit, return as is
        if len(tokens) <= max_tokens:
            return prompt
        
        # Truncate to max_tokens and decode back
        truncated_tokens = tokens[:max_tokens]
        truncated_prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        logger.info(f"🎯 Prompt truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
        
        return truncated_prompt


def test_prompt_enhancement():
    """Test function for prompt enhancement."""
    enhancer = PromptEnhancer()
    
    test_prompts = [
        "a sunset over mountains",
        "a cat playing in a garden",
        "a magical forest with glowing mushrooms"
    ]
    
    print("Testing GPT-2 Prompt Enhancement:")
    print("=" * 50)
    
    for prompt in test_prompts:
        print(f"\nOriginal: {prompt}")
        enhanced = enhancer.enhance_prompt(prompt, enhancement_type="diffusion")
        print(f"Enhanced: {enhanced}")
        print("-" * 30)


if __name__ == "__main__":
    test_prompt_enhancement()
