"""
Query decomposition and expansion service.
This module provides functionality to break down complex queries into simpler sub-queries
and generate alternative phrasings for improved information retrieval.
"""

from .bedrock import MLLM
from components.base_component import BaseComponent
from app.prompt import query_expansion_prompt
import ast


class Query_decomposer(BaseComponent):
    """
    Query decomposition and expansion handler.
    
    This class is responsible for:
    1. Breaking down complex queries into simpler sub-queries
    2. Generating alternative phrasings of queries
    3. Improving search accuracy through query expansion
    
    Attributes:
        queries (str): Storage for generated queries
        model (MLLM): Language model instance for query processing
    """

    def __init__(self):
        """Initialize the query decomposer with empty query storage."""
        super().__init__(logger_name='Query_decomposer')
        self.queries = ""
        self.model = MLLM()

    def run(self, query):
        """
        Process a query to generate expanded and decomposed versions.
        
        This method:
        1. Takes a complex query as input
        2. Uses the language model to generate alternative phrasings
        3. Parses the response into a list of queries
        4. Returns the expanded query set
        
        Args:
            query (str): The original query to process
            
        Returns:
            list: List of expanded and decomposed queries
            
        Note:
            The response from the language model is expected to be a valid Python list
            representation that can be parsed using ast.literal_eval.
        """
        # Generate expanded queries using the language model
        content = [{"type": "text", "text": query_expansion_prompt.format(query=query)}]
        self.queries = self.model.run(content)
        
        # Parse the response into a list of queries
        expanded_queries = ast.literal_eval(self.queries)
        self.logger.info(f'{expanded_queries}')
        
        return expanded_queries

