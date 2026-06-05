import os
import shutil
import numpy as np
from epochdb import EpochDB

test_dir = "./.test_debug_compact"
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

db = EpochDB(storage_dir=test_dir, dim=128)

# Write updates
db.remember("Temperature is 22.0 C", metadata={"triples": [("room1", "temperature", "22.0")]})
db.remember("Temperature is 25.0 C", metadata={"triples": [("room1", "temperature", "25.0")]})

mid = db.remember("To be deleted", metadata={"category": "temporary"})
db.delete(mid, hard=False)

db.force_checkpoint()

print("Epochs before compact:", db.cold_tier.get_all_epochs())
for ep in db.cold_tier.get_all_epochs():
    print(f"Epoch {ep} atoms:")
    for a in db.cold_tier.load_epoch(ep):
        print(f"  id={a.id}, payload_type={a.payload_type}, triples={a.triples}, metadata={a.metadata}")

db.compact()

print("\nEpochs after compact:", db.cold_tier.get_all_epochs())
for ep in db.cold_tier.get_all_epochs():
    print(f"Epoch {ep} atoms:")
    for a in db.cold_tier.load_epoch(ep):
        print(f"  id={a.id}, payload_type={a.payload_type}, triples={a.triples}, metadata={a.metadata}")

db.close()
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
