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
    (NODES, COMMODITIES, RUNTIME)
    Returns a list of tuples [(nodes, commodities, runtime), ...]
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # e.g. "(10000, 6, 12.62)" -> strip('()\n') -> "10000, 6, 12.62"
                nodes, commodities, runtime = line.strip('()\n').split(', ')
                data.append((int(nodes), int(commodities), float(runtime)))
    return data

def read_data_with_sets(file_path):
    """
    Parses a text file with multiple sets of runtime data.
    Returns a list of dictionaries for each set.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    sets = []
    current_set = {}
    for line in lines:
        line = line.strip()
        if ',' in line:
            parts = line.split(', ')
            if len(parts) == 2 and parts[0].isdigit():  # New set (iterations, threads)
                if current_set:  # Save the previous set
                    sets.append(current_set)
                current_set = {'iterations': int(parts[0]), 'threads': int(parts[1]), 'runtimes': {}}
            else:  # Runtime data
                method, runtime = parts
                current_set['runtimes'][method] = float(runtime)
    
    # Add the last set
    if current_set:
        sets.append(current_set)
    
    return sets

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
    """
    Plots a line graph comparing the performance gain (speedup) relative to ST,
    with x-axis as iterations and separate lines for each platform.
    """
    # Parse the file into sets
    sets = read_data_with_sets(file_path)
    
    # Prepare data for plotting
    performance_data = {}
    for data_set in sets:
        iterations = data_set['iterations']
        runtimes = data_set['runtimes']
        st_runtime = runtimes['ST']
        
        # Calculate performance gains for each method
        for method, runtime in runtimes.items():
            if method not in performance_data:
                performance_data[method] = {'iterations': [], 'gains': []}
            performance_data[method]['iterations'].append(iterations)
            performance_data[method]['gains'].append(st_runtime / runtime if method != 'ST' else 1.0)
    
    # Plot performance gain for each method
    plt.figure(figsize=(8, 5))
    for method, data in performance_data.items():
        plt.plot(data['iterations'], data['gains'], marker='o', linestyle='-', label=method)
    
    # Customize the graph
    plt.xlabel('Iterations')
    plt.ylabel('Performance Gain (Compared to ST)')
    plt.title('Performance Gain Comparison Across Platforms')
    plt.legend(title='Platforms')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate points
    for method, data in performance_data.items():
        for x, y in zip(data['iterations'], data['gains']):
            plt.text(x, y + 0.05, f'{y:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def prepare_individual_data_3(data_cuda, data_omp, data_st):
    """
    Combines and sorts data by (nodes, commodities) from three data lists:
      data_cuda, data_omp, data_st
    Each is a list of tuples [(nodes, commodities, runtime), ...].
    Returns:
      x_labels: list of "(nodes, commodities)" strings
      y_cuda, y_omp, y_st: lists of runtimes in corresponding order
    """
    # Collect all (nodes, commodities) pairs from all three data sets
    combined_data = sorted(set((n, c) for n, c, _ in data_cuda + data_omp + data_st))
    
    x_labels = [f"({n}, {c})" for n, c in combined_data]
    
    # For each (n,c), find the runtime in each data set
    y_cuda = [next((r for nn, cc, r in data_cuda if (nn, cc) == (n, c)), None) 
              for n, c in combined_data]
    y_omp  = [next((r for nn, cc, r in data_omp  if (nn, cc) == (n, c)), None) 
              for n, c in combined_data]
    y_st   = [next((r for nn, cc, r in data_st   if (nn, cc) == (n, c)), None) 
              for n, c in combined_data]
    
    return x_labels, y_cuda, y_omp, y_st

def plot_all_comparison_3(x_labels, y_cuda, y_omp, y_st):
    """
    Plots three lines (CUDA, OMP, ST) comparing runtimes across different 
    (nodes, commodities) pairs.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(x_labels, y_cuda, marker='^', label='CUDA', color='green')
    plt.plot(x_labels, y_omp,  marker='o', label='OMP',  color='blue')
    plt.plot(x_labels, y_st,   marker='s', label='Single Thread', color='orange')
    
    plt.xlabel('(Nodes, Commodities)')
    plt.ylabel('Runtime (s)')
    plt.title('Comparison of Runtime by (Nodes, Commodities)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
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
plot_all_comparison_3(x_labels, y_cuda, y_omp, y_st)
