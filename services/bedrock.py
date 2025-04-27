import boto3
from components.base_component import BaseComponent
from botocore.config import Config
from settings import settings
from app.config import config
import json
from enum import Enum

# Define your timeout settings (in seconds)
TIMEOUT = 900  # 15 mins

boto_config = Config(
    connect_timeout=TIMEOUT,
    read_timeout=TIMEOUT,
)


class Models(str, Enum):
    model1 = settings.MODEL_ID_SONNET_3_7


class MLLM(BaseComponent):

    def __init__(self):
        super().__init__(logger_name='MLLM')
        aws_settings = {
            'region_name': 'us-west-2',
            'config':boto_config,
            'service_name':'bedrock-runtime'
        }

        if config.IS_LOCAL == "True":
                aws_settings['aws_access_key_id'] = config.AWS_ACCESS_KEY_ID
                aws_settings['aws_secret_access_key'] = config.AWS_SECRET_ACCESS_KEY
        
        self.model_id = Models.model1
        self.client = boto3.client(**aws_settings)

    
    def run(self, data):

        llm_response = ""
        message_list = [
            {
                "role": 'user',
                "content": data
            }
        ]

        # Prepare the payload
        payload = json.dumps({
            "max_tokens": settings.MAX_TOKENS,
            'temperature': settings.TEMP,
            "anthropic_version": settings.ANTHROPIC_VERSION,
            "messages": message_list
        })

        try:
            response = self.client.invoke_model(
                body = payload,
                modelId = self.model_id
            )
            llm_response = json.loads(response.get("body").read())['content'][0]['text']
        except Exception as e:
             self.logger.info(f"An error occured while fetching the response from the llm")

        return llm_response