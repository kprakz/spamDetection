import requests 
# This script sends a POST request to the spam detection API and prints the response.


response = requests.post(
    "https://spamdetection-production.up.railway.app/predict",
    json={"subject": "Win a free iPhone today!"}
)
# Print the response from the API - Code == 200 means success
print("Response status code:", response.status_code)
print(response.json())