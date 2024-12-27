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

def calculate_performance_gain(data):
    """
    Calculates the performance gain compared to ST (Single-Thread) baseline.
    If ST is 15s and OMP is 10s, gain = 15/10 = 1.5, meaning 1.5x faster.
    Returns a dict { 'ST': 1.0, 'OMP': <gain>, 'CUDA': <gain>, ... }
    """
    baseline = data['ST']  # Use the ST value as the baseline
    return {key: baseline / value for key, value in data.items()}

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

def compare_performance(performance_gain_data):
    """
    Plots a bar chart comparing the performance gain (speedup) 
    relative to ST across different methods.
    """
    x = list(performance_gain_data.keys())
    y = list(performance_gain_data.values())

    plt.figure(figsize=(6, 4))
    plt.bar(x, y, color=['blue', 'orange', 'green'])
    plt.xlabel('Methods')
    plt.ylabel('Performance Gain (Compared to ST)')
    plt.title('Performance Gain Comparison')
    plt.ylim(0, max(y) + 0.1)
    # Annotate each bar with the performance gain value
    for i, v in enumerate(y):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
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

# --------------------------------------------------
# 1) Read the aggregate data for ST, OMP, CUDA
# --------------------------------------------------
data = read_data(cuda_omp_st_file)

# Calculate performance gain (speedup) with ST as the baseline
performance_gain = calculate_performance_gain(data)

# Visualize the raw runtimes in a bar chart
compare_runtime(data)

# Visualize the performance gain (speedup) in a bar chart
compare_performance(performance_gain)

# --------------------------------------------------
# 2) Read the individual (nodes, commodities, runtime) data from each file
#    Then plot a comparison line chart
# --------------------------------------------------
cuda_data = read_individual_data(cuda_file)
omp_data  = read_individual_data(omp_file)
st_data   = read_individual_data(st_file)

x_labels, y_cuda, y_omp, y_st = prepare_individual_data_3(cuda_data, omp_data, st_data)
plot_all_comparison_3(x_labels, y_cuda, y_omp, y_st)
