"""
AWS Bedrock integration service for the Multi-Modal RAG system.
This module provides functionality to interact with AWS Bedrock's language models
for generating responses to user queries.
"""

import boto3
from components.base_component import BaseComponent
from botocore.config import Config
from settings import settings
from app.config import config
import json
from enum import Enum

# Define timeout settings for AWS API calls (15 minutes)
TIMEOUT = 900

# Configure AWS client with timeout settings
boto_config = Config(
    connect_timeout=TIMEOUT,
    read_timeout=TIMEOUT,
)


class Models(str, Enum):
    """Enumeration of available language models."""
    model1 = settings.MODEL_ID_SONNET_3_7


class MLLM(BaseComponent):
    """
    AWS Bedrock Language Model interface.
    
    This class provides an interface to interact with AWS Bedrock's language models,
    specifically the Claude 3.7 Sonnet model. It handles:
    1. AWS client configuration and authentication
    2. Request formatting and model invocation
    3. Response processing and error handling
    
    Attributes:
        model_id (str): Identifier for the language model to use
        client: AWS Bedrock client instance
    """

    def __init__(self):
        """Initialize the MLLM with AWS Bedrock client configuration."""
        super().__init__(logger_name='MLLM')
        
        # Configure AWS client settings
        aws_settings = {
            'region_name': 'us-west-2',
            'config': boto_config,
            'service_name': 'bedrock-runtime'
        }

        # Add AWS credentials if running locally
        if config.IS_LOCAL == "True":
            aws_settings['aws_access_key_id'] = config.AWS_ACCESS_KEY_ID
            aws_settings['aws_secret_access_key'] = config.AWS_SECRET_ACCESS_KEY
        
        self.model_id = Models.model1
        self.client = boto3.client(**aws_settings)

    def run(self, data):
        """
        Generate a response using the language model.
        
        This method:
        1. Formats the input data into a message
        2. Prepares the request payload with model parameters
        3. Invokes the model through AWS Bedrock
        4. Processes and returns the response
        
        Args:
            data: Input data to send to the model. Can be a list of messages
                 with different content types (text, image)
            
        Returns:
            str: Generated response from the language model
            
        Note:
            The method handles errors gracefully and logs any issues that occur
            during the model invocation.
        """
        llm_response = ""
        
        # Format the input data as a user message
        message_list = [
            {
                "role": 'user',
                "content": data
            }
        ]

        # Prepare the request payload
        payload = json.dumps({
            "max_tokens": settings.MAX_TOKENS,
            'temperature': settings.TEMP,
            "anthropic_version": settings.ANTHROPIC_VERSION,
            "messages": message_list
        })

        try:
            # Invoke the model and process the response
            response = self.client.invoke_model(
                body=payload,
                modelId=self.model_id
            )
            llm_response = json.loads(response.get("body").read())['content'][0]['text']

        except Exception as e:
            self.logger.info(f"An error occurred while fetching the response from the llm")

        return llm_response