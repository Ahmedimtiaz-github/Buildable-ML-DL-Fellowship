import requests, json, os, sys, time

url = 'http://127.0.0.1:8000/agent'
data = {'question': 'best time to plant rice', 'top_k': '3', 'llm': 'local'}

# Try a simple form POST (no file)
try:
    r = requests.post(url, data=data, timeout=30)
    print('--- FORM POST (no file) ---')
    print('status:', r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text[:1000])
except Exception as e:
    print('FORM POST failed:', str(e))

# If there's a test image in the project root, try a multipart file upload
sample = os.path.join(os.getcwd(), 'test_leaf.jpg')
if os.path.exists(sample):
    try:
        with open(sample, 'rb') as fh:
            files = {'file': ('test_leaf.jpg', fh, 'image/jpeg')}
            r2 = requests.post(url, data=data, files=files, timeout=60)
            print('\\n--- FORM POST (with file test_leaf.jpg) ---')
            print('status:', r2.status_code)
            try:
                print(json.dumps(r2.json(), indent=2))
            except Exception:
                print(r2.text[:1000])
    except Exception as e:
        print('FILE upload test failed:', str(e))
else:
    print('\\nNo sample image test_leaf.jpg found in project root; skipping file-upload test.')
