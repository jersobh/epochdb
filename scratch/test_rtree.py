import rtree
p = rtree.index.Property()
p.dimension = 2
idx = rtree.index.Index(properties=p)

# [25, 25]
idx.insert(0, (25.0, -1e-9, 25.0, 1e-9))

# Query [20, 30]
hits = list(idx.intersection((20.0, -1e-9, 30.0, 1e-9)))
print(f"Hits for [20, 30]: {hits}")

# Query [25, 26]
hits = list(idx.intersection((25.0, -1e-9, 26.0, 1e-9)))
print(f"Hits for [25, 26]: {hits}")

# Query [26, 27]
hits = list(idx.intersection((26.0, -1e-9, 27.0, 1e-9)))
print(f"Hits for [26, 27]: {hits}")
