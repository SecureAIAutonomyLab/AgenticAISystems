# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ipython==8.32.0",
#     "langchain==0.3.17",
#     "langchain-community==0.3.16",
#     "langchain-openai==0.3.4",
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
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema
    from dotenv import load_dotenv, find_dotenv
    import marimo as mo
    return (
        IPython,
        LLMChain,
        OpenAI,
        PromptTemplate,
        ResponseSchema,
        StructuredOutputParser,
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
    working_dir = os.getcwd()
    status = load_dotenv(
        find_dotenv(
            filename=f'{working_dir}/tutorials/.env', 
            raise_error_if_not_found=True)
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
def _(ResponseSchema):
    # Define the expected output schema.
    response_schemas = [
        ResponseSchema(
            name="name", 
            description="The full name of the customer."
        ),
        ResponseSchema(
            name="email", 
            description="The email address of the customer."
        ),
        ResponseSchema(
            name="phone", 
            description="The phone number of the customer."
        ),
    ]
    return (response_schemas,)


@app.cell
def _(StructuredOutputParser, response_schemas):
    # Build the output parser from the response schemas.
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # Get formatting instructions that will tell the LLM how to format its output.
    format_instructions = output_parser.get_format_instructions()

    # Create a prompt template instructing the LLM to extract customer info and output in JSON format.
    template = """
    Extract the following customer information from the text:
    - Full name
    - Email address
    - Phone number

    Text:
    {text}

    {format_instructions}
    """
    return format_instructions, output_parser, template


@app.cell
def _(PromptTemplate, template):
    prompt = PromptTemplate(
        input_variables=["text", "format_instructions"],
        template=template,
    )
    return (prompt,)


@app.cell
def _(OpenAI):
    # Initialize the OpenAI LLM.
    llm = OpenAI(temperature=0.5)
    return (llm,)


@app.cell
def _(llm, prompt):
    # Compose the prompt and LLM using the chaining operator.
    chain = prompt | llm
    return (chain,)


@app.cell
def _(format_instructions):
    # Input text containing customer information.
    input_data = {
        "text": (
            "I recently ordered a new laptop, and I had a question. "
            "My name is Alice Johnson. Please contact me at alice.johnson@example.com "
            "or call me at (555) 123-4567."
        ),
        "format_instructions": format_instructions
    }
    return (input_data,)


@app.cell
def _(chain, input_data):
    # Execute the chain using the .invoke() method.
    raw_output = chain.invoke(input_data)
    print("Raw LLM output:")
    print(raw_output)
    return (raw_output,)


@app.cell
def _(output_parser, raw_output):
    # Parse the LLM's output string into a Python dictionary.
    parsed_output = output_parser.parse(raw_output)
    print("\nParsed output as Python dictionary:")
    print(parsed_output)
    return (parsed_output,)


@app.cell
def _(parsed_output):
    print(type(parsed_output))
    return


if __name__ == "__main__":
    app.run()
