#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "OMP_FlowAlgorithm.hpp"
#include "PropFlowAlgorithm.hpp"
#include "Commodity.hpp"
#include <iostream>
#include <fstream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/random.hpp>
#include <omp.h>

using namespace std;

using namespace boost;

// Define the graph type
Graph generate_random_graph(long long num_nodes, long long num_edges) {
    if (num_nodes <= 1) {
        throw std::invalid_argument("Number of nodes must be greater than 1.");
    }

    long long max_edges = num_nodes * (num_nodes - 1); // directed graph
    if (num_edges > max_edges) {
        std::cerr << "Requested number of edges (" << num_edges
            << ") exceeds the maximum possible edges (" << max_edges
            << ") for " << num_nodes << " nodes.\n";
        num_edges = max_edges; // adjust num_edges to the maximum
        std::cout << "Reducing number of edges to " << num_edges << ".\n";
    }

    Graph g(num_nodes);
    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist_weight(1, 20);  
    boost::random::uniform_int_distribution<> dist_capacity(10, 50); 

    long long edge_count = 0;

    cout << "Initializing edges..." << endl;
    // generate random edges
    while (edge_count < num_edges) {
        int u = gen() % num_nodes; //source
		int v = gen() % num_nodes; //destination

        if (u != v) {
            auto [edge, exists] = boost::edge(u, v, g);
            if (!exists) {
                // idd the edge if it doesn't exist
                auto e = boost::add_edge(u, v, g).first;
                g[e].capacity = dist_capacity(gen);  // set the capacity for the edge
                g[e].flow = 0;  // initialize the flow to 0

                put(boost::edge_weight, g, e, dist_weight(gen));
                edge_count++;
            }
        }
    }

    cout << "Graph with " << num_nodes << " nodes and " << num_edges << " edges created." << endl;

    return g;
}

Graph graph_test_init() {
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 0}, // 0 and 1
        {1, 2}, {2, 1}, // 1 and 2
        {2, 3}, {3, 2}, // 2 and 3
        {2, 5}, {5, 2}, // 2 and 5
        {4, 1}, {1, 4}, // 4 and 1
    };

    std::vector<EdgeProperties> edge_properties = {
        {10, 10}, {10, 10}, //  0 <-> 1
        {12, 20}, {12, 20}, //  1 <-> 2
        {8, 15}, {8, 15},   //  2 <-> 3
        {10, 40}, {10, 40}, //  2 <-> 5
        {15, 30}, {15, 30}, //  4 <-> 1
    };

    Graph g;

    for (size_t i = 0; i < edges.size(); ++i) {
        auto e = boost::add_edge(edges[i].first, edges[i].second, g).first;
        g[e].capacity = edge_properties[i].capacity;
		g[e].flow = edge_properties[i].flow;
		g[e].weight = edge_properties[i].weight;
    }

    return g;
}

void reset_flow_and_commodity(Graph& g, vector<Commodity> commodities) {
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        g[e].flow = 0;
    }

    for (Commodity c : commodities) {
        c.demand = c.init_demand;
        c.sent = 0;
    }
}

void init_OMP_n_single_thread_with_large_data(Graph g, vector<Commodity> commodities, double alpha, double epsilon) {
    //double avg_omp_rt = 0.0, avg_st_rt = 0.0;
    double omp_ratio, ori_ratio;
    double omp_rt = 0, ori_rt = 0;

    //write into file
    std::ofstream ompFile("..\\Python-TFS\\omp.txt");
    std::ofstream stFile("..\\Python-TFS\\st.txt");

    if (!ompFile && !stFile) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    omp_set_num_threads(8);
    for (int i = 0; i < 2; i++) {
        double omp_start = omp_get_wtime();
        omp_ratio = OMP_flowDistributionAlgorithm(g, commodities, epsilon, alpha);
        double omp_end = omp_get_wtime();
        reset_flow_and_commodity(g, commodities);
        omp_rt = omp_end - omp_start;

        cout << omp_rt << endl;
        ompFile << omp_rt << ", ";
    }
    //avg_omp_rt /= 3;

    for (int i = 0; i < 2; i++) {
        double ori_start = omp_get_wtime();
        ori_ratio = flowDistributionAlgorithm(g, commodities, epsilon, alpha);
        double ori_end = omp_get_wtime();
        reset_flow_and_commodity(g, commodities);
        ori_rt = ori_end - ori_start;

        cout << ori_rt << endl;
        stFile << ori_rt << ", ";
    }
    //avg_st_rt /= 3;

    // Close the file
    ompFile.close();
    stFile.close();
}

int main() {
    try {
        // Graph
        long long num_nodes = 250; // adjust nodes
        long long num_edges = 625000; // desired number of edges
        Graph g = graph_test_init();    
        //Graph g = generate_random_graph(num_nodes, num_edges);

        // Commodity
        int num_commodities = 20;   // number of commodities
        int min_demand = 10;        // minimum demand for a commodity
        int max_demand = 100;       // maximum demand for a commodity

        //std::vector<Commodity> commodities = generate_random_commodities(num_commodities, g, min_demand, max_demand);
	    std::vector<Commodity> commodities = {
		    {0, 3, 12},
		    {4, 5, 5},
	    };

        cout << "\n== Initial Commodities before Flow Distribution ==" << endl;
        for (const auto& commodity : commodities) {
            cout << "Commodity: Source = " << commodity.source
                << ", Destination = " << commodity.destination
                << ", Demand = " << commodity.demand << endl;
        }

        /*omp_set_num_threads(8);
        double omp_start = omp_get_wtime();
        double omp_ratio = OMP_flowDistributionAlgorithm(g, commodities, 0.01, 0.12);
        double omp_end = omp_get_wtime();

        reset_flow_and_commodity(g, commodities);*/

        double ori_start = omp_get_wtime();
        double ori_ratio = flowDistributionAlgorithm(g, commodities, 0.01, 0.12);
        double ori_end = omp_get_wtime();

        //double omp_runtime = omp_end - omp_start;
        double ori_runtime = ori_end - ori_start;

        //cout << "Max ratio (OMP): " << omp_ratio << endl;
        cout << "Max ratio (Original): " << ori_ratio << endl;
        cout << "Original Runtime: " << ori_runtime << endl;
        //cout << "OMP Runtime: " << omp_runtime << endl;

        // write into file
        std::ofstream mainFile("..\\Python-TFS\\cuda_omp_st.txt");
        std::ofstream ompFile("..\\Python-TFS\\omp.txt", std::ios::app);
        std::ofstream stFile("..\\Python-TFS\\st.txt", std::ios::app);

        if (!mainFile && !ompFile && !stFile) {
            std::cerr << "Error opening file!" << std::endl;
            return -1;
        }

        // write some text to the file
        mainFile << "ST, " << ori_runtime << std::endl;
        //mainFile << "OMP, " << omp_runtime << std::endl;

        //ompFile << "(" << boost::num_vertices(g) << ", " << commodities.size() << ", " << omp_runtime << ")" << endl;
        stFile << "(" << boost::num_vertices(g) << ", " << commodities.size() << ", " << ori_runtime << ")" << endl;

        // close the file
        mainFile.close();
        ompFile.close();
        stFile.close();

        // print all commodities sent
        cout << "\n== Commodities after Flow Distribution ==" << endl;
        for (const auto& commodity : commodities) {
            cout << "Commodity: Source = " << commodity.source
                << ", Destination = " << commodity.destination 
                << ", Demand = " << commodity.init_demand
                << ", Sent = " << commodity.sent << endl;
        }

    }
    catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << endl;
    }

    return 0;
}