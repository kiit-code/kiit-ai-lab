#!/usr/bin/env python
# uv: dependencies = ["requests"]

import requests

response = requests.get("https://astral.sh")
print(response.status_code)