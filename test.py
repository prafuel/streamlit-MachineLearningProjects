import requests

data = {
    "sms" : "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv"
}

url = "http://127.0.0.1:8000/"

res = requests.post(url=url, data=data)

print(res.text)