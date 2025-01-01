import matplotlib.pyplot as plt

def read_data(file_path):
    """
    Reads key-value pairs from a file with lines of the format:
    KEY, VALUE
    Returns a dictionary {KEY: VALUE}
    """
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                key, value = line.split(',')
                data[key.strip()] = float(value.strip())
    return data

def read_individual_data(file_path):
    """
    Reads lines of the format:
    (NUM_NODES, NUM_OF_COMMODITY, NUM_OF_ITER, RUNTIME)
    Returns a list of tuples [(num_nodes, num_of_commodity, num_of_iter, runtime), ...]
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # e.g. "(100000, 6, 10000, 11.3828)" -> strip('()\n') -> "100000, 6, 10000, 11.3828"
                num_nodes, num_of_commodity, num_of_iter, runtime = line.strip('()\n').split(', ')
                data.append((int(num_nodes), int(num_of_commodity), int(num_of_iter), float(runtime)))
    return data

def compare_runtime(data):
    """
    Plots a bar chart comparing the raw runtime of different platforms.
    data is a dict of format { 'ST': x, 'OMP': y, 'CUDA': z, ... }
    """
    x = list(data.keys())
    y = list(data.values())

    plt.figure(figsize=(6, 4))
    plt.bar(x, y, color=['blue', 'orange', 'green'])
    plt.xlabel('Platforms')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime of Different Parallel Computing Platforms')
    plt.tight_layout()
    plt.show()

def compare_performance_with_sets(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    metadata = None

    for line in lines:
        line = line.strip()
        if ',' in line and not any(char.isalpha() for char in line):  # Metadata line: no letters, just numbers
            # Parse metadata line
            metadata = tuple(map(int, line.split(',')))  # (nodes, commodities, iterations)
        elif line.startswith('ST') or line.startswith('OMP') or line.startswith('CUDA'):
            # Parse performance data
            method, runtime = line.split(',')
            runtime = float(runtime)
            
            # Extract thread count for OMP if available
            if 'OMP' in method:
                method_name, threads = method.split('(')
                threads = threads.strip(')')
                method = f"{method_name.strip()} ({threads} threads)"
            
            # Append parsed data with metadata
            data.append((metadata, method.strip(), runtime))

    # Organize data for plotting
    iterations = [d[0][2] for d in data if d[1] == 'ST']  # X-axis: Iterations
    methods = sorted(set(d[1] for d in data))  # Y-axis: Methods
    method_to_runtimes = {method: [] for method in methods}

    for iteration in iterations:
        for method in methods:
            runtime = next((d[2] for d in data if d[0][2] == iteration and d[1] == method), None)
            method_to_runtimes[method].append(runtime)

    # Plot the data
    plt.figure(figsize=(8, 6))
    for method, runtimes in method_to_runtimes.items():
        plt.plot(iterations, runtimes, marker='o', label=method)

    # Customize the plot
    plt.xticks(iterations)  # Show only iteration points on the X-axis
    plt.xlabel('Iterations')
    plt.ylabel('Runtime (seconds)')
    plt.title('Performance Comparison Across Methods')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate points with metadata
    for i, iteration in enumerate(iterations):
        for method, runtimes in method_to_runtimes.items():
            runtime = runtimes[i]
            if runtime is not None:  # Ensure runtime exists
                label = f"{metadata[0]}N, {metadata[1]}C"  # Nodes and Commodities
                plt.text(iteration, runtime + 0.1, label, ha='center', fontsize=8)


    plt.tight_layout()
    plt.show()

def prepare_individual_data_3(data_cuda, data_omp, data_st):
    """
    Combines and sorts data by num_of_iter from three data lists:
      data_cuda, data_omp, data_st
    Each is a list of tuples [(num_nodes, num_of_commodity, num_of_iter, runtime), ...].
    Returns:
      x_labels: list of "(iterations, num_nodes, num_of_commodity)" strings
      y_cuda, y_omp, y_st: lists of runtimes in corresponding order
    """
    # Collect all (num_nodes, num_of_commodity, num_of_iter) pairs from all three data sets
    combined_data = sorted(set((n, c, i) for n, c, i, _ in data_cuda + data_omp + data_st), key=lambda x: x[2])
    
    # Create labels with both iteration and (nodes, commodities)
    x_labels = [f"({i}, {n}, {c})" for n, c, i in combined_data]
    
    # For each (n,c), find the runtime in each data set
    y_cuda = [next((r for nn, cc, ii, r in data_cuda if (nn, cc, ii) == (n, c, i)), None) 
              for n, c, i in combined_data]
    y_omp  = [next((r for nn, cc, ii, r in data_omp  if (nn, cc, ii) == (n, c, i)), None) 
              for n, c, i in combined_data]
    y_st   = [next((r for nn, cc, ii, r in data_st   if (nn, cc, ii) == (n, c, i)), None) 
              for n, c, i in combined_data]
    
    return x_labels, y_cuda, y_omp, y_st

def plot_all_comparison_3(x_labels, y_cuda, y_omp, y_st, num_iterations):
    """
    Plots three lines (CUDA, OMP, ST) comparing runtimes across different 
    (nodes, commodities) pairs, including the number of iterations in the title.
    """
    plt.figure(figsize=(8, 5))  # Slightly larger figure size for better visibility

    # Plotting the data for each method
    plt.plot(x_labels, y_cuda, marker='^', label='CUDA', color='green', linestyle='-', markersize=6)
    plt.plot(x_labels, y_omp,  marker='o', label='OMP',  color='blue', linestyle='-', markersize=6)
    plt.plot(x_labels, y_st,   marker='s', label='Single Thread', color='orange', linestyle='-', markersize=6)
    
    plt.xlabel('(Iterations, Nodes, Commodities)', fontsize=12)
    plt.ylabel('Runtime (s)', fontsize=12)
    
    # Update the title to include the number of iterations
    plt.title(f'Comparison of Runtime by (Iterations, Nodes, Commodities) with {num_iterations} Iterations', fontsize=14)
    
    # Rotate x-axis labels to avoid overlap and align them more clearly
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Adjust the grid for better clarity
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add legend for clarity
    plt.legend(fontsize=10)
    
    # Tight layout to ensure no clipping of labels
    plt.tight_layout()
    plt.show()
# --------------------------------------------------
# File paths to the text files
# --------------------------------------------------
cuda_omp_st_file = 'cuda_omp_st.txt'  # Contains ST, OMP, CUDA runtimes
cuda_file        = 'cuda.txt'         # Contains data for CUDA alone
omp_file         = 'omp.txt'          # Contains data for OMP alone
st_file          = 'st.txt'           # Contains data for ST alone


# Visualize the performance gain (speedup) in a bar chart
compare_performance_with_sets(cuda_omp_st_file)

cuda_data = read_individual_data(cuda_file)
omp_data  = read_individual_data(omp_file)
st_data   = read_individual_data(st_file)

x_labels, y_cuda, y_omp, y_st = prepare_individual_data_3(cuda_data, omp_data, st_data)

num_iterations = st_data[0][2]

plot_all_comparison_3(x_labels, y_cuda, y_omp, y_st, num_iterations)
