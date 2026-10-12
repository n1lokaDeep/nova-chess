import inspect
from maia2 import inference, model
print("=== inference функции ===")
for n in dir(inference):
    if not n.startswith("_"):
        o = getattr(inference, n)
        if callable(o):
            try: print(f"  {n}{inspect.signature(o)}")
            except: print(f"  {n}")
print()
print("=== исходник prepare ===")
try:
    print(inspect.getsource(inference.prepare)[:800])
except Exception as e:
    print("нет prepare:", e)
