def analyze_weight_cn_file(filepath):
    """Analyze weight_cn.txt file to count non-zero weights per layer"""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    num_iterations = int(lines[0].strip())
    n_cols, n_rows = map(int, lines[1].strip().split())
    
    print(f"Number of iterations: {num_iterations}")
    print(f"Matrix dimensions: {n_cols} cols x {n_rows} rows")
    print()
    
    # Analyze each layer
    line_idx = 2
    layer_stats = []
    
    for iteration in range(num_iterations):
        print(f"=== Layer {iteration} ===")
        
        total_weights = 0
        nonzero_weights = 0
        weight_values = []
        
        # Process each check node (row) in this iteration
        for row in range(n_rows):
            if line_idx < len(lines):
                weights = list(map(float, lines[line_idx].strip().split()))
                line_idx += 1
                
                # Count non-zero weights (excluding padding zeros)
                row_nonzeros = sum(1 for w in weights if abs(w) > 1e-10)
                nonzero_weights += row_nonzeros
                total_weights += len(weights)
                
                # Collect non-zero weight values for statistics
                weight_values.extend([w for w in weights if abs(w) > 1e-10])
        
        # Calculate statistics
        sparsity = (total_weights - nonzero_weights) / total_weights * 100
        
        stats = {
            'iteration': iteration,
            'total_weights': total_weights,
            'nonzero_weights': nonzero_weights,
            'sparsity_percent': sparsity,
            'mean_weight': np.mean(weight_values) if weight_values else 0,
            'std_weight': np.std(weight_values) if weight_values else 0,
            'min_weight': np.min(np.abs(weight_values)) if weight_values else 0,
            'max_weight': np.max(np.abs(weight_values)) if weight_values else 0
        }
        
        layer_stats.append(stats)
        
        print(f"Total weight positions: {total_weights}")
        print(f"Non-zero weights: {nonzero_weights}")
        print(f"Sparsity: {sparsity:.2f}%")
        print(f"Weight statistics:")
        print(f"  Mean: {stats['mean_weight']:.6f}")
        print(f"  Std:  {stats['std_weight']:.6f}")
        print(f"  Range: [{stats['min_weight']:.6f}, {stats['max_weight']:.6f}]")
        print()
    
    return layer_stats

def plot_layer_analysis(layer_stats):
    """Plot weight statistics across layers"""
    import matplotlib.pyplot as plt
    
    iterations = [s['iteration'] for s in layer_stats]
    nonzero_counts = [s['nonzero_weights'] for s in layer_stats]
    sparsities = [s['sparsity_percent'] for s in layer_stats]
    mean_weights = [s['mean_weight'] for s in layer_stats]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Non-zero weight count per layer
    ax1.bar(iterations, nonzero_counts)
    ax1.set_title('Non-zero Weights per Layer')
    ax1.set_xlabel('Iteration/Layer')
    ax1.set_ylabel('Count')
    
    # Sparsity per layer
    ax2.plot(iterations, sparsities, 'o-')
    ax2.set_title('Sparsity per Layer')
    ax2.set_xlabel('Iteration/Layer')
    ax2.set_ylabel('Sparsity (%)')
    
    # Mean weight magnitude per layer
    ax3.plot(iterations, mean_weights, 's-')
    ax3.set_title('Mean Weight Magnitude per Layer')
    ax3.set_xlabel('Iteration/Layer')
    ax3.set_ylabel('Mean Weight')
    
    # Weight distribution across all layers
    all_weights = []
    for stats in layer_stats:
        # You'd need to modify the analysis function to return actual weight values
        pass
    
    plt.tight_layout()
    plt.show()

# Usage
layer_stats = analyze_weight_cn_file("path/to/weight_cn.txt")
plot_layer_analysis(layer_stats)
