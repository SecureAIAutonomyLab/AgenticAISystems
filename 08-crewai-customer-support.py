# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "crewai==0.102.0",
#     "crewai-tools==0.36.0",
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
    mo.md(
        r"""
        # L3: Multi-agent Customer Support Automation

        In this lesson, you will learn about the six key elements which help make Agents perform even better:
        - Role Playing
        - Focus
        - Tools
        - Cooperation
        - Guardrails
        - Memory
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
    mo.md(r"""- Import libraries, API and LLM""")
    return


@app.cell
def _():
    from crewai import Agent, Task, Crew
    return Agent, Crew, Task


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
    mo.md(r"""## Role Playing, Focus and Cooperation""")
    return


@app.cell
def _(Agent):
    support_agent = Agent(
        role="Senior Support Representative",
    	goal="Be the most friendly and helpful "
            "support representative in your team",
    	backstory=(
    		"You work at crewAI (https://crewai.com) and "
            " are now working on providing "
    		"support to {customer}, a super important customer "
            " for your company."
    		"You need to make sure that you provide the best support!"
    		"Make sure to provide full complete answers, "
            " and make no assumptions."
    	),
    	allow_delegation=False,
    	verbose=True
    )
    return (support_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - By not setting `allow_delegation=False`, `allow_delegation` takes its default value of being `True`.
        - This means the agent _can_ delegate its work to another agent which is better suited to do a particular task.
        """
    )
    return


@app.cell
def _(Agent):
    support_quality_assurance_agent = Agent(
    	role="Support Quality Assurance Specialist",
    	goal="Get recognition for providing the "
        "best support quality assurance in your team",
    	backstory=(
    		"You work at crewAI (https://crewai.com) and "
            "are now working with your team "
    		"on a request from {customer} ensuring that "
            "the support representative is "
    		"providing the best support possible.\n"
    		"You need to make sure that the support representative "
            "is providing full"
    		"complete answers, and make no assumptions."
    	),
    	verbose=True
    )
    return (support_quality_assurance_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        * **Role Playing**: Both agents have been given a role, goal and backstory.
        * **Focus**: Both agents have been prompted to get into the character of the roles they are playing.
        * **Cooperation**: Support Quality Assurance Agent can delegate work back to the Support Agent, allowing for these agents to work together.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Tools, Guardrails and Memory

        ### Tools
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Import CrewAI tools""")
    return


@app.cell
def _():
    from crewai_tools import SerperDevTool, \
                             ScrapeWebsiteTool, \
                             WebsiteSearchTool
    return ScrapeWebsiteTool, SerperDevTool, WebsiteSearchTool


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Possible Custom Tools
        - Load customer data
        - Tap into previous conversations
        - Load data from a CRM
        - Checking existing bug reports
        - Checking existing feature requests
        - Checking ongoing tickets
        - ... and more
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - Some ways of using CrewAI tools.

        ```Python
        search_tool = SerperDevTool()
        scrape_tool = ScrapeWebsiteTool()
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - Instantiate a document scraper tool.
        - The tool will scrape a page (only 1 URL) of the CrewAI documentation.
        """
    )
    return


@app.cell
def _(ScrapeWebsiteTool):
    docs_scrape_tool = ScrapeWebsiteTool(
        website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
    )
    return (docs_scrape_tool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##### Different Ways to Give Agents Tools

        - Agent Level: The Agent can use the Tool(s) on any Task it performs.
        - Task Level: The Agent will only use the Tool(s) when performing that specific Task.

        **Note**: Task Tools override the Agent Tools.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Creating Tasks
        - You are passing the Tool on the Task Level.
        """
    )
    return


@app.cell
def _(Task, docs_scrape_tool, support_agent):
    inquiry_resolution = Task(
        description=(
            "{customer} just reached out with a super important ask:\n"
    	    "{inquiry}\n\n"
            "{person} from {customer} is the one that reached out. "
    		"Make sure to use everything you know "
            "to provide the best support possible."
    		"You must strive to provide a complete "
            "and accurate response to the customer's inquiry."
        ),
        expected_output=(
    	    "A detailed, informative response to the "
            "customer's inquiry that addresses "
            "all aspects of their question.\n"
            "The response should include references "
            "to everything you used to find the answer, "
            "including external data or solutions. "
            "Ensure the answer is complete, "
    		"leaving no questions unanswered, and maintain a helpful and friendly "
    		"tone throughout."
        ),
    	tools=[docs_scrape_tool],
        agent=support_agent,
    )
    return (inquiry_resolution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - `quality_assurance_review` is not using any Tool(s)
        - Here the QA Agent will only review the work of the Support Agent
        """
    )
    return


@app.cell
def _(Task, support_quality_assurance_agent):
    quality_assurance_review = Task(
        description=(
            "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
            "Ensure that the answer is comprehensive, accurate, and adheres to the "
    		"high-quality standards expected for customer support.\n"
            "Verify that all parts of the customer's inquiry "
            "have been addressed "
    		"thoroughly, with a helpful and friendly tone.\n"
            "Check for references and sources used to "
            " find the information, "
    		"ensuring the response is well-supported and "
            "leaves no questions unanswered."
        ),
        expected_output=(
            "A final, detailed, and informative response "
            "ready to be sent to the customer.\n"
            "This response should fully address the "
            "customer's inquiry, incorporating all "
    		"relevant feedback and improvements.\n"
    		"Don't be too formal, we are a chill and cool company "
    	    "but maintain a professional and friendly tone throughout."
        ),
        agent=support_quality_assurance_agent,
    )
    return (quality_assurance_review,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Creating the Crew

        #### Memory
        - Setting `memory=True` when putting the crew together enables Memory.
        """
    )
    return


@app.cell
def _(
    Crew,
    inquiry_resolution,
    quality_assurance_review,
    support_agent,
    support_quality_assurance_agent,
):
    crew = Crew(
      agents=[support_agent, support_quality_assurance_agent],
      tasks=[inquiry_resolution, quality_assurance_review],
      verbose=True,
      memory=True
    )
    return (crew,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Running the Crew

        **Note**: LLMs can provide different outputs for they same input, so what you get might be different than what you see in the video.

        #### Guardrails
        - By running the execution below, you can see that the agents and the responses are within the scope of what we expect from them.
        """
    )
    return


@app.cell
def _(crew):
    inputs = {
        "customer": "DeepLearningAI",
        "person": "Andrew Ng",
        "inquiry": "I need help with setting up a Crew "
                   "and kicking it off, specifically "
                   "how can I add memory to my crew? "
                   "Can you provide guidance?"
    }
    result = crew.kickoff(inputs=inputs)
    return inputs, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Display the final result as Markdown.""")
    return


@app.cell
def _(result):
    result.raw
    return


@app.cell
def _(mo, result):
    mo.md(result.raw)
    return


if __name__ == "__main__":
    app.run()
