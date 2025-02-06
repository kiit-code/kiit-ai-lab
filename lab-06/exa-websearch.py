# pip install exa-py
from exa_py import Exa
exa = Exa(api_key = "6178919e-dcd9-4eed-8fdb-ce11fd0a1198")
result = exa.search_and_contents(
    "find blog posts about AGI",
    text = { "max_characters": 1000 }
)
print(result)