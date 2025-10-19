# scripts/heartbeat_smoke.py
"""
Quick smoke: pings GPT-5 (dry prompt) and Alpaca (account/status).
Run: python scripts/heartbeat_smoke.py
Requires your usual env vars for OpenAI + Alpaca.
"""
import os, time
from infra.heartbeat import wrap_heartbeat

# Optional: you can swap this to your actual clients.
try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

import requests


@wrap_heartbeat("openai","chat.completions", req_meta_fn=lambda **k: {"model":k.get("model")})
def ping_gpt(model="gpt-5", messages=None, **opts):
    if client is None: raise RuntimeError("OpenAI client not available")
    messages = messages or [{"role":"user","content":"Health check. Reply 'pong'."}]
    # Medium reasoning effort
    return client.chat.completions.create(model=model, messages=messages, reasoning={'effort':'medium'}, **opts)


@wrap_heartbeat("alpaca","v2/account", method="GET")
def ping_alpaca_account():
    base = os.getenv("ALPACA_BASE_URL","https://paper-api.alpaca.markets")
    key = os.getenv("APCA_API_KEY_ID"); sec = os.getenv("APCA_API_SECRET_KEY")
    r = requests.get(f"{base}/v2/account", headers={"APCA-API-KEY-ID":key or "", "APCA-API-SECRET-KEY":sec or ""}, timeout=15)
    class Resp: pass
    resp = Resp(); resp.status_code = r.status_code; resp.usage = {}
    if r.status_code >= 400: raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:120]}")
    return resp


if __name__ == "__main__":
    try:
        ping_gpt()
        print("GPT-5 OK")
    except Exception as e:
        print("GPT-5 FAIL:", e)
    try:
        ping_alpaca_account()
        print("Alpaca OK")
    except Exception as e:
        print("Alpaca FAIL:", e)
    print("Tail last 20:")
    from infra.heartbeat import tail_heartbeat
    tail_heartbeat(20, True)
