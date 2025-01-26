# from phi.agent import Agent
# from phi.tools.python import PythonTools

# agent = Agent(tools=[PythonTools()], show_tool_calls=True)
# agent.print_response("Write a python script for fibonacci series and display the result till the 10th number")

from phi.agent import Agent, RunResponse
from phi.model.groq import Groq

agent = Agent(model=Groq(id="mixtral-8x7b-32768"), markdown=True)

# Get the response in a variable
# run: RunResponse = agent.run("Share a 2 sentence horror story.")
# print(run.content)

# Print the response in the terminal
agent.print_response("You are a code agent. Output a complete program in C to add two numbers.")
