import requests, os
url = "http://127.0.0.1:8000/agent"
data = {"question":"is this leaf diseased?","top_k":"3","llm":"local"}
fname = "test_leaf.jpg"
files = {"file": open(fname,"rb")} if os.path.exists(fname) else None
r = requests.post(url, data=data, files=files)
print("STATUS:", r.status_code)
print("RESPONSE:", r.text)
