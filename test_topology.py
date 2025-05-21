def test_topology_generation():
    """Test topology generation functions."""
    n_in = 4
    n_hidden = 64
    n_out = 2
    seed = 42
    
    # Compute baseline weight budget
    n_w_fc = (n_in + n_hidden) * n_hidden + (n_hidden + 1) * n_out
    target_n_weights = int(n_w_fc)
    
    # Test FC topology
    adj_fc, meta_fc = make_fc(n_in, n_hidden, n_out)
    assert adj_fc.shape == (n_in + n_hidden + n_out, n_in + n_hidden + n_out)
    assert meta_fc['type'] == 'fc'
    assert int(adj_fc.sum()) == target_n_weights
    
    # Test random topology
    adj_rs, meta_rs = make_random(n_in, n_hidden, n_out, density=0.1, seed=seed,
                                target_n_weights=target_n_weights)
    assert adj_rs.shape == (n_in + n_hidden + n_out, n_in + n_hidden + n_out)
    assert meta_rs['type'] == 'random'
    assert int(adj_rs.sum()) == target_n_weights
    
    # Test modular topology
    adj_mod, meta_mod = make_modular(n_in, n_hidden, n_out, density=0.1, seed=seed,
                                   target_n_weights=target_n_weights)
    assert adj_mod.shape == (n_in + n_hidden + n_out, n_in + n_hidden + n_out)
    assert meta_mod['type'] == 'modular'
    assert int(adj_mod.sum()) == target_n_weights
    
    # Test Watts-Strogatz topology
    adj_ws, meta_ws = make_ws(n_in, n_hidden, n_out, k=4, p=0.1, seed=seed,
                             target_n_weights=target_n_weights)
    assert adj_ws.shape == (n_in + n_hidden + n_out, n_in + n_hidden + n_out)
    assert meta_ws['type'] == 'ws'
    assert int(adj_ws.sum()) == target_n_weights
    
    # Test structural properties
    for adj, meta in [(adj_fc, meta_fc), (adj_rs, meta_rs), 
                      (adj_mod, meta_mod), (adj_ws, meta_ws)]:
        # Check weight counts
        assert meta['n_weights'] == target_n_weights
        assert int(adj.sum()) == target_n_weights
        
        # Check fan-in statistics
        assert 'fan_in_hidden_mean' in meta
        assert 'fan_in_hidden_std' in meta
        assert 'fan_in_out_mean' in meta
        assert 'fan_in_out_std' in meta
        
        # Check minimum degrees
        for i in range(n_in + n_hidden + n_out):
            assert adj[:, i].sum() >= 2  # min in-degree
            assert adj[i, :].sum() >= 2  # min out-degree
        
        # Verify padding was applied last
        adj_np = adj.cpu().numpy()
        assert adj_np.sum() == target_n_weights, "Weight count mismatch after padding"
        
        # Check sparsity is consistent
        sparsity = 1.0 - (meta['n_weights'] / (n_in + n_hidden + n_out) ** 2)
        assert abs(sparsity - meta.get('sparsity', sparsity)) < 1e-6, "Sparsity mismatch" 