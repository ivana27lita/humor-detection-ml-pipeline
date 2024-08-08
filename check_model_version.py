import requests
from pprint import PrettyPrinter

pp = PrettyPrinter()
response = requests.get("http://localhost:8080/v1/models/humor-detection-model")
pp.pprint(response.json())
