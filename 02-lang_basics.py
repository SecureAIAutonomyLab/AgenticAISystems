# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ipython==8.32.0",
#     "langchain==0.3.17",
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
    import marimo as mo
    import openai
    import os
    import IPython
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from dotenv import load_dotenv, find_dotenv
    import marimo as mo
    return (
        IPython,
        LLMChain,
        OpenAI,
        PromptTemplate,
        find_dotenv,
        load_dotenv,
        mo,
        openai,
        os,
    )


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
            raise_error_if_not_found=True
        )
    )

    # API configuration
    client = openai.OpenAI(api_key='')
    client.api_key = os.getenv("OPENAI_API_KEY")

    # LangChain
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
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


@app.cell
def _(OpenAI, PromptTemplate):
    # Define a prompt template.

    # Here, we're asking the LLM to explain a topic in simple terms.

    prompt = PromptTemplate(

        input_variables=["topic"],

        template="Explain the concept of {topic} in simple terms."

    )

    # Initialize the OpenAI LLM.

    # Make sure your environment variable OPENAI_API_KEY is set with your OpenAI API key.

    llm = OpenAI(temperature=0.5)

    # Compose the prompt and LLM using the chaining operator.

    # This creates a RunnableSequence equivalent to the old LLMChain.

    chain = prompt | llm

    # Prepare the input as a dictionary.

    input_data = {"topic": "quantum computing"}

    # Execute the chain using the invoke() method.

    result = chain.invoke(input_data)
    return chain, input_data, llm, prompt, result


@app.cell
def _(result):
    print("Response from the OpenAI LLM:")
    print(result)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
