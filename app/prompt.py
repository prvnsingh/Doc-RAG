summary_prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""

summary_prompt_image = """
Describe the image in detail. For context, the images are extracted from a book of Mechanical Engineering
"""

retriever_prompt = """

"""

user_query_prompt = """
Answer the question based only on the following context, which can include text, tables, and the below image.
Context: {context_text}
Question: {user_question}
"""
