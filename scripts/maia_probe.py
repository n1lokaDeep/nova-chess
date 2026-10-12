import maia2, inspect
print("=== API maia2 ===")
print([x for x in dir(maia2) if not x.startswith("_")])
print()
from maia2 import inference, model
print("inference:", [x for x in dir(inference) if not x.startswith("_")])
print("model:", [x for x in dir(model) if not x.startswith("_")])
print()
m = model.from_pretrained(type="rapid", device="cpu")
print("модель загружена:", type(m))
