import requests

scheme_code = "119551"

url = f"https://api.mfapi.in/mf/{scheme_code}"

response = requests.get(url, timeout=15)
response.raise_for_status()

data = response.json()

meta = data.get("meta", {})

print("Scheme Name:", meta.get("scheme_name"))
print("Fund House:", meta.get("fund_house"))
print("Scheme Type:", meta.get("scheme_type"))
print("Scheme Category:", meta.get("scheme_category"))