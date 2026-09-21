import base64, json, sys, time, urllib.request
img = sys.argv[1] if len(sys.argv) > 1 else "/root/test-capture.png"
q = sys.argv[2] if len(sys.argv) > 2 else "Décris cette capture d'écran en 3 phrases : que voit-on, et y a-t-il un problème visible ?"
b = base64.b64encode(open(img, "rb").read()).decode()
body = {"model": "flash-next", "max_tokens": 1500, "messages": [{"role": "user", "content": [{"type": "text", "text": q}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}}]}]}
req = urllib.request.Request("http://127.0.0.1:8086/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    d = json.load(urllib.request.urlopen(req, timeout=300)); m = d["choices"][0]["message"]
    print("réponse en %.1f s, usage %s" % (time.time() - t0, d["usage"]))
    print("clés :", list(m.keys())); print("reasoning :", ((m.get("reasoning_content") or m.get("reasoning") or ""))[:300].replace("\n", " "))
    print("content   :", (m.get("content") or "")[:900])
except urllib.error.HTTPError as e:
    print("ERREUR HTTP", e.code, e.read().decode()[:500])
except Exception as e:
    print("ERREUR", e)
