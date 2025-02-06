import os
from textwrap import dedent

# --- Environment Variable Check ---
if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY environment variable is not set. "
        "Please set GROQ_API_KEY before running the script."
    )

# Import necessary modules.
from agno.agent import Agent
from agno.models.groq import Groq  # Using Groq instead of OpenAIChat
from agno.tools.exa import ExaTools

# Initialize the shopping partner agent with the Groq model.
agent = Agent(
    name="shopping partner",
    model=Groq(id="deepseek-r1-distill-llama-70b"),  # Replace with your preferred Groq model ID if needed.
    instructions=[
        "You are a product recommender agent specializing in finding products that match user preferences.",
        "Prioritize finding products that satisfy as many user requirements as possible, but ensure a minimum match of 50%.",
        "Search for products only from authentic and trusted e-commerce websites such as Amazon, Flipkart, Myntra, Meesho, Google Shopping, Nike, and other reputable platforms.",
        "Verify that each product recommendation is in stock and available for purchase.",
        "Avoid suggesting counterfeit or unverified products.",
        "Clearly mention the key attributes of each product (e.g., price, brand, features) in the response.",
        "Format the recommendations neatly and ensure clarity for ease of user understanding.",
    ],
    tools=[ExaTools()],
    show_tool_calls=True,
)

# Print the agent's response based on the user query.
agent.print_response(
    "I am looking for running shoes with the following preferences: Color: Black, Purpose: Comfortable for long-distance running, Budget: Under Rs. 10,000"
)