from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
# from phi.tools.newspaper4k import Newspaper4k

from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()], #Newspaper4k()], #remove newspaper4k
    description="You are a senior NYT researcher writing an article on a topic.",
    instructions="Always include the sources",
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    # debug_mode=True,
)
agent.print_response("Tata Safari", stream=True)
