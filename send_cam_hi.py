# import requests

# ESP32_URL = "http://192.168.0.250/cam-hi.jpg"
# API_URL   = "http://127.0.0.1:8000/analyze"

# age = 8

# # 1) download image
# img = requests.get(ESP32_URL, timeout=10)
# img.raise_for_status()

# # 2) upload to VLM server
# files = {"image": ("snap.jpg", img.content, "image/jpeg")}
# data = {"age": str(age)}

# r = requests.post(API_URL, files=files, data=data, timeout=120)
# print("Status:", r.status_code)
# print(r.text)
import requests

ESP32_URL = "http://192.168.0.250/cam-hi.jpg"   # snapshot endpoint
API_URL   = "http://127.0.0.1:8000/analyze"     # your VLM server

age = 8
rotate = 0  # try 90/270 if image is rotated

# 1) Download one JPEG snapshot from ESP32
resp = requests.get(ESP32_URL, timeout=10)
resp.raise_for_status()  # stops if not 200 OK

jpeg_bytes = resp.content
print("Downloaded", len(jpeg_bytes), "bytes from ESP32")

# 2) Upload that JPEG to your FastAPI server
files = {
    "image": ("snap.jpg", jpeg_bytes, "image/jpeg")
}
data = {
    "age": str(age),
    "rotate": str(rotate),
    "force_enhance": "1",
}

r = requests.post(API_URL, files=files, data=data, timeout=180)
print("Server status:", r.status_code)
print(r.text)
