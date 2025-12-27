from app import app

client = app.test_client()
resp = client.post('/', data={'feedback_text': 'Great workplace\nToo many meetings\nGood salary and benefits'})
print('Status code:', resp.status_code)
text = resp.get_data(as_text=True)
print('Response length:', len(text))
with open('smoke_test_response.html', 'w', encoding='utf-8') as f:
    f.write(text[:5000])
print('Saved snippet to smoke_test_response.html')
