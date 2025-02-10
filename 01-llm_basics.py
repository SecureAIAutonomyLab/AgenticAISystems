# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "openai==1.61.1",
#     "python-dotenv==1.0.1",
# ]
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from dotenv import load_dotenv, find_dotenv
    import openai
    import os
    return find_dotenv, load_dotenv, openai, os


@app.cell
def _(os):
    # Delete if not using in UTSA 
    os.environ["http_proxy"] = "http://xa-proxy.utsarr.net:80"
    os.environ["https_proxy"] = "http://xa-proxy.utsarr.net:80"
    return


@app.cell
def _(find_dotenv, load_dotenv, openai, os):
    # Ensure you have a .env file in the AgenticAISystems folder with your OPENAI_API_KEY.
    working_dir = os.getcwd()
    status = load_dotenv(
        find_dotenv(
            filename=f'{working_dir}/AgenticAISystems/.env', 
            raise_error_if_not_found=True)
    )

    client = openai.OpenAI(api_key='')
    client.api_key  = os.environ['OPENAI_API_KEY']
    return client, status, working_dir


@app.cell
def _(client):
    def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature # this is the degree of randomness of the model's output
        )
        print(str(response.choices[0].message))
        return response.choices[0].message.content
    return get_completion, get_completion_from_messages


@app.cell
def _(get_completion):
    _prompt = f"""
    what is the capital of France ?
    """
    _response = get_completion(_prompt)
    print(_response)
    return


@app.cell
def _(get_completion):
    _prompt = f"""
    Remember my name is peyman najafirad.
    """
    _response = get_completion(_prompt)
    print(_response)
    return


@app.cell
def _(get_completion):
    _prompt = f"""
    what is my name?
    """
    _response = get_completion(_prompt)
    print(_response)
    return


@app.cell
def _(get_completion_from_messages):
    _system_message = f"""
    You are a friendly fitness coach who gives simple, encouraging, \
    and practical health advice. Keep your answers positive, easy to \
    follow, and supportive. Avoid technical jargon and focus on motivation.
    """
    _user_message = f"""\
    What is the best way to stay healthy?"""

    _messages =  [
    {'role':'system',
     'content': _system_message},
    {'role':'user',
     'content': f"{_user_message}"},
    ]
    _response = get_completion_from_messages(_messages, model="gpt-3.5-turbo", temperature=0)
    print(_response)
    return


@app.cell
def _(get_completion_from_messages):
    _system_message = f"""
    You are a wise and caring grandparent who shares health advice in a \
    warm and simple way. Your responses should feel like a comforting conversation \
    with a loved one, full of wisdom and kindness.
    """
    _user_message = f"""\
    What is the best way to stay healthy?"""

    _messages =  [
    {'role':'system',
     'content': _system_message},
    {'role':'user',
     'content': f"{_user_message}"},
    ]
    _response = get_completion_from_messages(_messages, model="gpt-3.5-turbo", temperature=0)
    print(_response)
    return


@app.cell
def _(get_completion_from_messages):
    _delimiter = "####"
    _system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with \
    {_delimiter} characters.
    Classify each query into a primary category \
    and a secondary category.
    Provide your output in json format with the \
    keys: primary and secondary.

    Primary categories: Billing, Technical Support, \
    Account Management, or General Inquiry.

    Billing secondary categories:
    Unsubscribe or upgrade
    Add a payment method
    Explanation for charge
    Dispute a charge

    Technical Support secondary categories:
    General troubleshooting
    Device compatibility
    Software updates

    Account Management secondary categories:
    Password reset
    Update personal information
    Close account
    Account security

    General Inquiry secondary categories:
    Product information
    Pricing
    Feedback
    Speak to a human

    """
    _user_message = f"""\
    I want you to delete my profile and all of my user data"""
    _messages =  [
    {'role':'system',
     'content': _system_message},
    {'role':'user',
     'content': f"{_delimiter}{_user_message}{_delimiter}"},
    ]

    _response = get_completion_from_messages(_messages, model="gpt-3.5-turbo", temperature=0)
    print(_response)
    return


if __name__ == "__main__":
    app.run()
