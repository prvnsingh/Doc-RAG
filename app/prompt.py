"""
Prompt templates for the Multi-Modal RAG system.
This module contains all the prompt templates used for different tasks in the application,
including summarization, retrieval, query decomposition, and query expansion.
"""

# Prompt template for summarizing text and tables
summary_prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""

# Prompt template for describing images
summary_prompt_image = """
Describe the image in detail.
"""

# Prompt template for answering user questions based on retrieved context
user_query_prompt = """
You are a helpful assistant. Provide a JSON Answer for question based on the provided context.
Do not rely on the internal memory or internal knowledge.
Determine if the question is relevant to the context. 
If yes, answer the question using the context provided and return status = 1 with answer. 
If no, return status = 0 and answer something like this:
\"I\'m sorry, I don't have enough information to answer that.\"

Context: {context_text} 
Question: {user_question}

Respond in this JSON format:
{{\"status\":...,\"answer\":\"...\"}}

Only provide the JSON.
"""

# Prompt template for query expansion and decomposition
query_expansion_prompt = """
You are a helpful assistant that rewrites a user query to improve search accuracy in a document retrieval system.

Given an input query, you have two task
 task 1 : Your task is to decompose a complex or multi-part question into simpler, atomic sub-questions that can be answered individually.
        Make sure each sub-question:
        - Targets a single fact or concept.
        - Preserves the intent and context of the original query.
        - Can be used independently to search for relevant documents.

 task 2: generate alternative phrasings or related queries that capture similar intent, using different vocabulary or structure. Do not change the meaning.

Original Query:
"{query}"

Generated queries: []

Generate only a python list containing 5-6 queries, NO OTHER TEXT.
"""
