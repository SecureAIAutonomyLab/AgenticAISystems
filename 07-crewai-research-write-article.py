# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "crewai==0.102.0",
#     "httpx==0.28.1",
#     "ipython==8.32.0",
#     "marimo",
#     "python-dotenv==1.0.1",
#     "utils==1.0.2",
# ]
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can download the `requirements.txt` for this course from the workspace of this lab. `File --> Open...`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # L2: Create Agents to Explore and Write an Article

        In this lesson, you will be introduced to the foundational concepts of multi-agent systems and get an overview of the crewAI framework.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The libraries are already installed in the classroom. If you're running this notebook on your own machine, you can install the following:
        ```Python
        !pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29
        ```
        """
    )
    return


@app.cell
def _():
    # Warning control
    import warnings
    warnings.filterwarnings('ignore')
    return (warnings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Import from the crewAI libray.""")
    return


@app.cell
def _():
    from crewai import Agent, Task, Crew
    return Agent, Crew, Task


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - As a LLM for your agents, you'll be using OpenAI's `gpt-3.5-turbo`.

        **Optional Note:** crewAI also allow other popular models to be used as a LLM for your Agents. You can see some of the examples at the [bottom of the notebook](#1).
        """
    )
    return


@app.cell
def _(os):
    # Delete if not using in UTSA 
    os.environ["http_proxy"] = "http://xa-proxy.utsarr.net:80"
    os.environ["https_proxy"] = "http://xa-proxy.utsarr.net:80"
    return


@app.cell
def _():
    import os
    from dotenv import load_dotenv, find_dotenv

    working_dir = os.getcwd()
    status = load_dotenv(
        find_dotenv(
            filename=f'{working_dir}/AgenticAISystems/.env', 
            raise_error_if_not_found=True
        )
    )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

    # Disable sending telemetry data
    os.environ["OTEL_SDK_DISABLED"] = "true"
    return find_dotenv, load_dotenv, openai_api_key, os, status, working_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating Agents

        - Define your Agents, and provide them a `role`, `goal` and `backstory`.
        - It has been seen that LLMs perform better when they are role playing.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Agent: Planner

        **Note**: The benefit of using _multiple strings_ :
        ```Python
        varname = "line 1 of text"
                  "line 2 of text"
        ```

        versus the _triple quote docstring_:
        ```Python
        varname = \"\"\"line 1 of text
                     line 2 of text
                  \"\"\"
        ```
        is that it can avoid adding those whitespaces and newline characters, making it better formatted to be passed to the LLM.
        """
    )
    return


@app.cell
def _(Agent):
    planner = Agent(
        role="Content Planner",
        goal="Plan engaging and factually accurate content on {topic}",
        backstory="You're working on planning a blog article "
                  "about the topic: {topic}."
                  "You collect information that helps the "
                  "audience learn something "
                  "and make informed decisions. "
                  "Your work is the basis for "
                  "the Content Writer to write an article on this topic.",
        allow_delegation=False,
    	verbose=True
    )
    return (planner,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Agent: Writer""")
    return


@app.cell
def _(Agent):
    writer = Agent(
        role="Content Writer",
        goal="Write insightful and factually accurate "
             "opinion piece about the topic: {topic}",
        backstory="You're working on a writing "
                  "a new opinion piece about the topic: {topic}. "
                  "You base your writing on the work of "
                  "the Content Planner, who provides an outline "
                  "and relevant context about the topic. "
                  "You follow the main objectives and "
                  "direction of the outline, "
                  "as provide by the Content Planner. "
                  "You also provide objective and impartial insights "
                  "and back them up with information "
                  "provide by the Content Planner. "
                  "You acknowledge in your opinion piece "
                  "when your statements are opinions "
                  "as opposed to objective statements.",
        allow_delegation=False,
        verbose=True
    )
    return (writer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Agent: Editor""")
    return


@app.cell
def _(Agent):
    editor = Agent(
        role="Editor",
        goal="Edit a given blog post to align with "
             "the writing style of the organization. ",
        backstory="You are an editor who receives a blog post "
                  "from the Content Writer. "
                  "Your goal is to review the blog post "
                  "to ensure that it follows journalistic best practices,"
                  "provides balanced viewpoints "
                  "when providing opinions or assertions, "
                  "and also avoids major controversial topics "
                  "or opinions when possible.",
        allow_delegation=False,
        verbose=True
    )
    return (editor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating Tasks

        - Define your Tasks, and provide them a `description`, `expected_output` and `agent`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Task: Plan""")
    return


@app.cell
def _(Task, planner):
    plan = Task(
        description=(
            "1. Prioritize the latest trends, key players, "
                "and noteworthy news on {topic}.\n"
            "2. Identify the target audience, considering "
                "their interests and pain points.\n"
            "3. Develop a detailed content outline including "
                "an introduction, key points, and a call to action.\n"
            "4. Include SEO keywords and relevant data or sources."
        ),
        expected_output="A comprehensive content plan document "
            "with an outline, audience analysis, "
            "SEO keywords, and resources.",
        agent=planner,
    )
    return (plan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Task: Write""")
    return


@app.cell
def _(Task, writer):
    write = Task(
        description=(
            "1. Use the content plan to craft a compelling "
                "blog post on {topic}.\n"
            "2. Incorporate SEO keywords naturally.\n"
    		"3. Sections/Subtitles are properly named "
                "in an engaging manner.\n"
            "4. Ensure the post is structured with an "
                "engaging introduction, insightful body, "
                "and a summarizing conclusion.\n"
            "5. Proofread for grammatical errors and "
                "alignment with the brand's voice.\n"
        ),
        expected_output="A well-written blog post "
            "in markdown format, ready for publication, "
            "each section should have 2 or 3 paragraphs.",
        agent=writer,
    )
    return (write,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Task: Edit""")
    return


@app.cell
def _(Task, editor):
    edit = Task(
        description=("Proofread the given blog post for "
                     "grammatical errors and "
                     "alignment with the brand's voice."),
        expected_output="A well-written blog post in markdown format, "
                        "ready for publication, "
                        "each section should have 2 or 3 paragraphs.",
        agent=editor
    )
    return (edit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating the Crew

        - Create your crew of Agents
        - Pass the tasks to be performed by those agents.
            - **Note**: *For this simple example*, the tasks will be performed sequentially (i.e they are dependent on each other), so the _order_ of the task in the list _matters_.
        - `verbose=2` allows you to see all the logs of the execution.
        """
    )
    return


@app.cell
def _(Crew, edit, editor, plan, planner, write, writer):
    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan, write, edit],
        verbose=True
    )
    return (crew,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Running the Crew""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Note**: LLMs can provide different outputs for they same input, so what you get might be different than what you see in the video.""")
    return


@app.cell
def _(crew):
    result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Display the results of your execution as markdown in the notebook.""")
    return


@app.cell
def _(mo, result):
    mo.md(result.raw.replace("```", ""))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Try it Yourself

        - Pass in a topic of your choice and see what the agents come up with!
        """
    )
    return


@app.cell
def _(crew):
    topic = 'How does CrewAI Work?'
    result_1 = crew.kickoff(inputs={'topic': topic})
    return result_1, topic


@app.cell
def _(result_1):
    result_1
    return


@app.cell
def _(mo, result_1):
    mo.md(result_1.raw)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Other Popular Models as LLM for your Agents""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Hugging Face (HuggingFaceHub endpoint)

        ```Python
        from langchain_community.llms import HuggingFaceHub

        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token="<HF_TOKEN_HERE>",
            task="text-generation",
        )

        ### you will pass "llm" to your agent function
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Mistral API

        ```Python
        OPENAI_API_KEY=your-mistral-api-key
        OPENAI_API_BASE=https://api.mistral.ai/v1
        OPENAI_MODEL_NAME="mistral-small"
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Cohere

        ```Python
        from langchain_community.chat_models import ChatCohere
        # Initialize language model
        os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
        llm = ChatCohere()

        ### you will pass "llm" to your agent function
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### For using Llama locally with Ollama and more, checkout the crewAI documentation on [Connecting to any LLM](https://docs.crewai.com/how-to/LLM-Connections/).""")
    return


if __name__ == "__main__":
    app.run()
