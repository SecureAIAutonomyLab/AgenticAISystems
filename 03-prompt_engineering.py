# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ipython==8.32.0",
#     "langchain==0.3.17",
#     "langchain-community==0.3.16",
#     "marimo",
#     "openai==1.61.1",
#     "python-dotenv==1.0.1",
# ]
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Getting Started with Prompt Engineering

        This notebook contains examples and exercises to learning about prompt engineering.

        We will be using the [OpenAI APIs](https://platform.openai.com/) for all examples. I am using the default settings `temperature=0.7` and `top-p=1`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""---""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Prompt Engineering Basics

        Objectives
        - Load the libraries
        - Review the format
        - Cover basic prompts
        - Review common use cases
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Below we are loading the necessary libraries, utilities, and configurations.""")
    return


@app.cell
def _():
    import openai
    import os
    import IPython
    from langchain_community.llms import OpenAI
    from dotenv import load_dotenv, find_dotenv
    import marimo as mo
    return IPython, OpenAI, find_dotenv, load_dotenv, mo, openai, os


@app.cell
def _(os):
    # Delete if not using in UTSA 
    os.environ["http_proxy"] = "http://xa-proxy.utsarr.net:80"
    os.environ["https_proxy"] = "http://xa-proxy.utsarr.net:80"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Load environment variables. You can use anything you like but I used `python-dotenv`. Just create a `.env` file with your `OPENAI_API_KEY` in the AgenticAISystems folder then load it.""")
    return


@app.cell
def _(find_dotenv, load_dotenv, openai, os):
    working_dir = os.getcwd()
    status = load_dotenv(
        find_dotenv(
            filename=f'{working_dir}/AgenticAISystems/.env', 
            raise_error_if_not_found=True)
    )

    # API configuration
    client = openai.OpenAI(api_key='')
    client.api_key = os.getenv("OPENAI_API_KEY")
    return client, status, working_dir


@app.cell
def _(client):
    def set_open_params(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        """ set openai parameters"""

        openai_params = {}    

        openai_params['model'] = model
        openai_params['temperature'] = temperature
        openai_params['max_tokens'] = max_tokens
        openai_params['top_p'] = top_p
        openai_params['frequency_penalty'] = frequency_penalty
        openai_params['presence_penalty'] = presence_penalty
        return openai_params

    def get_completion(params, messages):
        """ GET completion from openai api"""

        response = client.chat.completions.create(
            model = params['model'],
            messages = messages,
            temperature = params['temperature'],
            max_tokens = params['max_tokens'],
            top_p = params['top_p'],
            frequency_penalty = params['frequency_penalty'],
            presence_penalty = params['presence_penalty'],
        )
        return response
    return get_completion, set_open_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Basic prompt example:""")
    return


@app.cell
def _(get_completion, set_open_params):
    params = set_open_params()
    _prompt = 'The sky is blue and'
    messages = [{'role': 'user', 'content': _prompt}]
    response = get_completion(params, messages)
    return messages, params, response


@app.cell
def _(response):
    response.choices[0].message.content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Try with different temperature to compare results:""")
    return


@app.cell
def _(IPython, get_completion, messages, set_open_params):
    params_1 = set_open_params(temperature=0)
    response_1 = get_completion(params_1, messages)
    IPython.display.Markdown(response_1.choices[0].message.content)
    return params_1, response_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1.1 Text Summarization""")
    return


@app.cell
def _(IPython, get_completion, set_open_params):
    params_2 = set_open_params(temperature=0.7)
    _prompt = "Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body's immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance. \n\nExplain the above in one sentence:"
    messages_1 = [{'role': 'user', 'content': _prompt}]
    response_2 = get_completion(params_2, messages_1)
    IPython.display.Markdown(response_2.choices[0].message.content)
    return messages_1, params_2, response_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Exercise: Instruct the model to explain the paragraph in one sentence like "I am 5". Do you see any differences?""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1.2 Question Answering""")
    return


@app.cell
def _(IPython, get_completion, params_2):
    _prompt = 'Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.\n\nContext: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.\n\nQuestion: What was OKT3 originally sourced from?\n\nAnswer:'
    messages_2 = [{'role': 'user', 'content': _prompt}]
    response_3 = get_completion(params_2, messages_2)
    IPython.display.Markdown(response_3.choices[0].message.content)
    return messages_2, response_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Context obtained from here: https://www.nature.com/articles/d41586-023-00400-x""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Exercise: Edit prompt and get the model to respond that it isn't sure about the answer.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1.3 Text Classification""")
    return


@app.cell
def _(IPython, get_completion, params_2):
    _prompt = 'Classify the text into neutral, negative or positive.\n\nText: I think the food was okay.\n\nSentiment:'
    messages_3 = [{'role': 'user', 'content': _prompt}]
    response_4 = get_completion(params_2, messages_3)
    IPython.display.Markdown(response_4.choices[0].message.content)
    return messages_3, response_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Exercise: Modify the prompt to instruct the model to provide an explanation to the answer selected.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1.4 Role Playing""")
    return


@app.cell
def _(IPython, get_completion, params_2):
    _prompt = 'The following is a conversation with an AI research assistant. The assistant tone is technical and scientific.\n\nHuman: Hello, who are you?\nAI: Greeting! I am an AI research assistant. How can I help you today?\nHuman: Can you tell me about the creation of blackholes?\nAI:'
    messages_4 = [{'role': 'user', 'content': _prompt}]
    messages_4 = [{'role': 'user', 'content': _prompt}]
    response_5 = get_completion(params_2, messages_4)
    IPython.display.Markdown(response_5.choices[0].message.content)
    return messages_4, response_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Exercise: Modify the prompt to instruct the model to keep AI responses concise and short.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1.5 Code Generation""")
    return


@app.cell
def _(IPython, get_completion, params_2):
    _prompt = '"""\nTable departments, columns = [DepartmentId, DepartmentName]\nTable students, columns = [DepartmentId, StudentId, StudentName]\nCreate a MySQL query for all students in the Computer Science Department\n"""'
    messages_5 = [{'role': 'user', 'content': _prompt}]
    response_6 = get_completion(params_2, messages_5)
    IPython.display.Markdown(response_6.choices[0].message.content)
    return messages_5, response_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1.6 Reasoning""")
    return


@app.cell
def _(IPython, get_completion, params_2):
    _prompt = 'The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. \n\nSolve by breaking the problem into steps. First, identify the odd numbers, add them, and indicate whether the result is odd or even.'
    messages_6 = [{'role': 'user', 'content': _prompt}]
    response_7 = get_completion(params_2, messages_6)
    IPython.display.Markdown(response_7.choices[0].message.content)
    return messages_6, response_7


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Exercise: Improve the prompt to have a better structure and output format.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Advanced Prompting Techniques

        Objectives:

        - Cover more advanced techniques for prompting: few-shot, chain-of-thoughts,...
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.2 Few-shot prompts""")
    return


@app.cell
def _(IPython, get_completion, params_2):
    _prompt = '''The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
    A: The answer is False.

    The odd numbers in this group add up to an even number: 17,  10, 19, 4, 8, 12, 24.
    A: The answer is True.

    The odd numbers in this group add up to an even number: 16,  11, 14, 4, 8, 13, 24.
    A: The answer is True.

    The odd numbers in this group add up to an even number: 17,  9, 10, 12, 13, 4, 2.
    A: The answer is False.

    The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
    A:'''
    messages_7 = [{'role': 'user', 'content': _prompt}]
    response_8 = get_completion(params_2, messages_7)
    IPython.display.Markdown(response_8.choices[0].message.content)
    return messages_7, response_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.3 Chain-of-Thought (CoT) Prompting""")
    return


@app.cell
def _(IPython, get_completion, params_2):
    _prompt = 'The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.\nA: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.\n\nThe odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. \nA:'
    messages_8 = [{'role': 'user', 'content': _prompt}]
    response_9 = get_completion(params_2, messages_8)
    IPython.display.Markdown(response_9.choices[0].message.content)
    return messages_8, response_9


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.4 Zero-shot CoT""")
    return


@app.cell
def _(IPython, get_completion, params_2):
    _prompt = "I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?\n\nLet's think step by step."
    messages_9 = [{'role': 'user', 'content': _prompt}]
    response_10 = get_completion(params_2, messages_9)
    IPython.display.Markdown(response_10.choices[0].message.content)
    return messages_9, response_10


if __name__ == "__main__":
    app.run()
