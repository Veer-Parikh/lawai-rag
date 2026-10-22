import sys
import time

print("Importing health...", flush=True)
t0 = time.time()
import api.routers.health
print(f"health: {time.time()-t0:.2f}s", flush=True)

print("Importing chat...", flush=True)
t0 = time.time()
import api.routers.chat
print(f"chat: {time.time()-t0:.2f}s", flush=True)

print("Importing search...", flush=True)
t0 = time.time()
import api.routers.search
print(f"search: {time.time()-t0:.2f}s", flush=True)

print("Importing admin...", flush=True)
t0 = time.time()
import api.routers.admin
print(f"admin: {time.time()-t0:.2f}s", flush=True)
