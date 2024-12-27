import matplotlib.pyplot as plt


def read_data(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                key, value = line.split(',')
                data[key.strip()] = float(value.strip())
    return data

def read_individual_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                nodes, commodities, runtime = line.strip('()\n').split(', ')
                data.append((int(nodes), int(commodities), float(runtime)))
    return data

def calculate_performance_gain(data):
    baseline = data['ST']  # Use the ST value as the baseline
    return {key: baseline / value  for key, value in data.items()}

def prepare_individual_data(data1, data2):
    # combine and sort data by nodes and commodities
    combined_data = sorted(set((n, c) for n, c, _ in data1 + data2))
    
    # extract runtimes for each (nodes, commodities) pair
    x_labels = [f"({n}, {c})" for n, c in combined_data]
    y1 = [next((r for n1, c1, r in data1 if (n1, c1) == (n, c)), None) for n, c in combined_data]
    y2 = [next((r for n2, c2, r in data2 if (n2, c2) == (n, c)), None) for n, c in combined_data]
    
    return x_labels, y1, y2

# Function to plot the data as a bar chart
def compare_runtime(data):
    x = list(data.keys())
    y = list(data.values())

    plt.bar(x, y, color=['blue', 'orange'])
    plt.xlabel('Platforms')
    plt.ylabel('Runtime')
    plt.title('Runtime of different Parallel Computing Platforms')
    plt.show()

def compare_performance(performance_gain_data):
    x = list(performance_gain_data.keys())
    y = list(performance_gain_data.values())

    plt.bar(x, y, color=['blue', 'orange'])
    plt.xlabel('Methods')
    plt.ylabel('Performance Gain (Compared to ST)')
    plt.title('Performance Gain Comparison')
    plt.ylim(0, max(y) + 0.1)  
    for i, v in enumerate(y):  
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    plt.show()

def plot_all_comparison(x, y1, y2):
    plt.plot(x, y1, marker='o', label='OMP', color='blue')
    plt.plot(x, y2, marker='s', label='Single Thread', color='orange')
    plt.xlabel('Nodes, Commodities (Sorted by Nodes)')
    plt.ylabel('Runtime (s)')
    plt.title('Comparison of Runtime by (Nodes, Commodities)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout for better fit
    plt.show()

# File path to the text file
omp_st_file = 'omp_st.txt'
omp_file = 'omp.txt'
st_file = 'st.txt'

# Read the data and plot it
data = read_data(omp_st_file)
performance_gain = calculate_performance_gain(data)
compare_runtime(data)
compare_performance(performance_gain)

omp_data = read_individual_data(omp_file)
st_data = read_individual_data(st_file)
x_labels, y1, y2 = prepare_individual_data(omp_data, st_data)
plot_all_comparison(x_labels, y1, y2)

