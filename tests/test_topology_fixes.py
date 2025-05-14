from topologies.modular import make_modular
from topologies.random_sparse import make_rs

def test_modular():
    print("\nTesting Modular topology:")
    mask, meta = make_modular(
        n_in=4,
        n_hidden=64,
        n_out=2,
        n_modules=6,
        p_intra=0.8,
        p_inter=0.010,
        density=0.10,
        seed=0
    )
    
    print("\nResults:")
    print(f"Core density: {meta['density']:.3f}")
    print(f"Core clustering: {meta['avg_clustering']:.3f}")
    print(f"Modularity Q: {meta['modularity']:.3f}")
    
    # Validate network requirements
    assert 0.085 <= meta['density'] <= 0.115, f"Density {meta['density']:.3f} outside target range [0.085, 0.115]"
    assert meta['modularity'] >= 0.60, f"Modularity {meta['modularity']:.3f} below target 0.60"
    print("✓ Modular topology meets requirements")

def test_random_sparse():
    print("\nTesting Random Sparse topology:")
    mask, meta = make_rs(
        n_in=4,
        n_hidden=64,
        n_out=2,
        density=0.10,
        seed=0
    )
    
    print("\nResults:")
    print(f"Core density: {meta['density']:.3f}")
    print(f"Core clustering: {meta['avg_clustering']:.3f}")
    
    # Validate network requirements
    assert 0.085 <= meta['density'] <= 0.115, f"Density {meta['density']:.3f} outside target range [0.085, 0.115]"
    print("✓ Random Sparse topology meets requirements")

if __name__ == "__main__":
    test_modular()
    test_random_sparse()
    print("\nAll tests passed!") 