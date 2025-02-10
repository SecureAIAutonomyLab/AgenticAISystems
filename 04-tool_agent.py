# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "python-dotenv==1.0.1",
#     "langchain==0.3.17",
#     "langchain-openai==0.3.4",
#     "langchain-core==0.3.34",
# ]
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Building a Tool-Calling Agent with LangChain

        - Define simple math tools (addition, multiplication, exponentiation).
        - Configure a language model and prompt template.
        - Create and execute a tool-calling agent that performs a complex arithmetic operation.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import os
    from dotenv import load_dotenv, find_dotenv
    from langchain import hub
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder
    )
    return (
        AgentExecutor,
        ChatOpenAI,
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
        SystemMessagePromptTemplate,
        create_tool_calling_agent,
        find_dotenv,
        hub,
        load_dotenv,
        mo,
        os,
        tool,
    )


@app.cell
def _(os):
    # Delete or modify these lines if you are not behind the UTSA proxy.
    os.environ["http_proxy"] = "http://xa-proxy.utsarr.net:80"
    os.environ["https_proxy"] = "http://xa-proxy.utsarr.net:80"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Load Environment Variables
     
        Create a `.env` file in the `AgenticAISystems/` folder and add your key:

            ```OPENAI_API_KEY=<your-openai-key>```
        """
    )
    return


@app.cell
def _(find_dotenv, load_dotenv, os):
    # Ensure you have a .env file with your OPENAI_API_KEY.
    working_dir = os.getcwd()
    status = load_dotenv(
        find_dotenv(
            filename=f'{working_dir}/AgenticAISystems/.env', 
            raise_error_if_not_found=True
        )
    )
    return status, working_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Define Tools

        Define three simple math tools using the `@tool` decorator:

        - **multiply**: Multiplies two integers.
        - **add**: Adds two integers.
        - **exponentiate**: Raises a base to a given exponent.

        These tools are later made available for the agent to call.
        """
    )
    return


@app.cell
def _(tool):
    @tool
    def multiply(first_int: int, second_int: int) -> int:
        """Multiply two integers together."""
        return first_int * second_int

    @tool
    def add(first_int: int, second_int: int) -> int:
        "Add two integers."
        return first_int + second_int

    @tool
    def exponentiate(base: int, exponent: int) -> int:
        "Exponentiate the base to the exponent power."
        return base**exponent
    return add, exponentiate, multiply


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Configure the Language Model and Prompt Template

        - Initialize a `ChatOpenAI` model (using GPT-3.5-turbo-0125).
        - Create a chat prompt template with:
            - A system message defining the assistant as "helpful".
            - A human message that takes user input.
            - A placeholder (`agent_scratchpad`) for the intermediate steps after each tool is called.
        """
    )
    return


@app.cell
def _(
    ChatOpenAI,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    chat_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    chat_prompt.pretty_print()
    return chat_prompt, llm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Build and Execute the Tool-Calling Agent

        - Compile our defined math tools into a list.
        - Create a tool-calling agent by combining the language model, tools, and prompt.
        - An `AgentExecutor` is used to run the agent with verbose output.
        """
    )
    return


@app.cell
def _(
    AgentExecutor,
    add,
    chat_prompt,
    create_tool_calling_agent,
    exponentiate,
    llm,
    multiply,
):
    _tools = [multiply, add, exponentiate]

    # Construct the tool calling agent
    _agent = create_tool_calling_agent(llm, _tools, chat_prompt)

    # Create an agent executor by passing in the agent and tools
    _agent_executor = AgentExecutor(agent=_agent, tools=_tools, verbose=True)

    _results = _agent_executor.invoke(
        {
            "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
        }
    )

    _results
    return


if __name__ == "__main__":
    app.run()
