# check_gemini_models.py (robust version)
import os, traceback, sys
from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
except Exception as e:
    print("google.genai package not available:", e)
    raise

GEMINI_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GENAI_API_KEY")
)
if not GEMINI_KEY:
    raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY/GENAI_API_KEY) in env before running.")

client = genai.Client(api_key=GEMINI_KEY)

def summarize_model_obj(m):
    """Try to extract useful fields from a model object in a safe way."""
    # Common attributes across SDKs
    fields = {}
    for attr in ("name", "id", "display_name", "description", "model", "model_id"):
        try:
            val = getattr(m, attr)
        except Exception:
            val = None
        if val:
            fields[attr] = val

    # If object is dict-like
    try:
        if isinstance(m, (dict,)):
            for k in ("name", "id", "display_name", "description"):
                if k in m and m[k]:
                    fields[k] = m[k]
    except Exception:
        pass

    # fallback: try .to_dict() or .__dict__
    try:
        if not fields:
            if hasattr(m, "to_dict"):
                d = m.to_dict()
                for k in ("name", "id", "display_name", "description"):
                    if d.get(k):
                        fields[k] = d.get(k)
            elif hasattr(m, "__dict__"):
                d = vars(m)
                for k in ("name", "id", "display_name", "description"):
                    if d.get(k):
                        fields[k] = d.get(k)
    except Exception:
        pass

    # final fallback: repr
    if not fields:
        fields["repr"] = repr(m)[:300]

    return fields

def print_models_iterable(models_iter, limit=50):
    count = 0
    try:
        for m in models_iter:
            count += 1
            print(f"\n=== Model #{count} ===")
            info = summarize_model_obj(m)
            for k, v in info.items():
                print(f"{k}: {v}")
            if count >= limit:
                print(f"...reached limit of {limit} displayed models.")
                break
    except Exception:
        print("Failed while iterating models_iter:")
        traceback.print_exc()

print("Attempting known model-listing entrypoints...")

tried = []
# candidate callables / patterns to try in order
candidates = [
    ("client.list_models()", lambda: client.list_models()),
    ("client.models.list()", lambda: client.models.list() if hasattr(client, "models") and hasattr(client.models, "list") else None),
    ("client.models.list_models()", lambda: client.models.list_models() if hasattr(client, "models") and hasattr(client.models, "list_models") else None),
    ("client.models()", lambda: getattr(client, "models")()),
    ("client.list()", lambda: getattr(client, "list")()),
]

found = False
for name, fn in candidates:
    try:
        tried.append(name)
        models = fn()
        if models is None:
            continue

        print(f"\nUsing entrypoint: {name}")
        # If models is a simple list/tuple
        if isinstance(models, (list, tuple)):
            print("-> returned a list/tuple with", len(models), "entries.")
            print_models_iterable(models)
            found = True
            break

        # If models is an iterable/pager (common case)
        if hasattr(models, "__iter__") and not isinstance(models, (str, bytes, dict)):
            print("-> returned an iterable/pager object. Iterating...")
            print_models_iterable(models)
            found = True
            break

        # If models has .models or .data attributes
        if hasattr(models, "models"):
            print("-> has .models attribute; inspecting...")
            print_models_iterable(models.models)
            found = True
            break
        if hasattr(models, "data"):
            print("-> has .data attribute; inspecting...")
            print_models_iterable(models.data)
            found = True
            break

        # fallback: just print repr
        print("-> Unknown return type; printing repr:")
        print(type(models))
        print(repr(models)[:1000])
        found = True
        break

    except Exception:
        print(f"entrypoint {name} failed:")
        traceback.print_exc()

if not found:
    print("\nTried the following entrypoints but none returned models:", tried)
    print("\nTips / next steps:")
    print(" - Verify the API key has correct permissions and is not expired.")
    print(" - Check network connectivity / proxy issues.")
    print(" - Try upgrading the google-genai package: pip install --upgrade google-genai")
    print(" - Print client object: print(client) and inspect available attributes (dir(client)).")
