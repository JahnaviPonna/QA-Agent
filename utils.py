# backend/utils.py
import json
def safe_json_parse(s):
    try:
        return json.loads(s)
    except:
        return None
