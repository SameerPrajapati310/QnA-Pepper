prompt_template = """"
You are an expert at creating question based on DBMS material and documentation.
Your goal is to prepare a coder or programer for their exam and coding tests.
You do this by asking questions about text below:
---------
{text}
---------
create ten question that will prepare the coders or programers for their tests.
Make sure not to lose any important information.

QUESTION:
"""


refine_template = ("""
        You are an expert at creating question based on DBMS material and documentation.
        Your goal is to prepare a coder or programer for their exam and coding tests.
        We have received some practical questions to a certain extent : {existing_answer}.
        we have the option to refine the existing question or add new ones.
        {text}
        prepare ten question 
        Given the new context, refine the original question in English.
        If the content is not helpful, please provide the original question.
        QUESTION:
""")