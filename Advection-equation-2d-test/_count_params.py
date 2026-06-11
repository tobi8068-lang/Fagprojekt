def count(layers):
    return sum(r*c + c for r, c in layers)

for in_dim, label in [(22, "with RFF"), (3, "no RFF")]:
    v   = count([(in_dim,64),(64,64),(64,64),(64,64),(64,1)])
    e34 = count([(in_dim,34),(34,34),(34,34),(34,34),(34,1)])
    e36 = count([(in_dim,36),(36,36),(36,36),(36,36),(36,1)])
    g   = count([(in_dim,16),(16,3)])
    m34 = 3*e34 + g
    m36 = 3*e36 + g
    print(f"\n--- {label} (in_dim={in_dim}) ---")
    print(f"  Vanilla (w=64)       : {v:>7,}")
    print(f"  MoE 3x(w=34) + gate  : {m34:>7,}  ({100*(v-m34)/v:+.1f}%)")
    print(f"  MoE 3x(w=36) + gate  : {m36:>7,}  ({100*(v-m36)/v:+.1f}%)")
