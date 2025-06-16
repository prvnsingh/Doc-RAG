"""
Nemo Guardrails integration service for content safety and moderation.
This module provides functionality to interact with Nvidia's Nemo Guardrails
for content safety checks and moderation.
"""
from nemoguardrails import RailsConfig
from nemoguardrails import LLMRails
from settings import settings
from components.base_component import BaseComponent
import os
import yaml
import json
import random

class GuardrailsService(BaseComponent):
    """
    Nemo Guardrails service for content safety and moderation.
    
    This class provides an interface to interact with Nvidia's Nemo Guardrails
    for content safety checks and moderation. It handles:
    1. Content safety checks for input and output
    2. Integration with AWS Bedrock
    3. Response processing and error handling
    """

    def __init__(self):
        """Initialize the Guardrails service with configuration."""
        super().__init__(logger_name='Guardrails')


        # Create the Rails configuration
        content_safety_config = RailsConfig.from_path('./config')

        # Initialize the Rails
        self.rails = LLMRails(content_safety_config)

    def run(self, content):
        """
        Run the Guardrails service to check content safety.

        This method processes the input content through the Guardrails service
        and returns the validated response.

        Args:
            content (str): The input content to be checked for safety.

        Returns:
            str: The validated response from the Guardrails service.
        """
        # Format the input data as a user message
        message_list = [
            {
                "role": 'user',
                "content": content
            }
        ]

        try:
            response = self.rails.generate(messages=message_list)
            self.logger.info(f"Guardrails response: {response}")
            return response['content']
        except Exception as e:
            self.logger.error(f"Guardrails error: {e}")
            return None