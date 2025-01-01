#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "OMP_FlowAlgorithm.hpp"
#include "FlowAlgorithm.hpp"
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

    long long max_edges = num_nodes * (num_nodes - 1);
    if (num_edges > max_edges) {
        std::cerr << "Requested number of edges (" << num_edges
            << ") exceeds the maximum possible edges (" << max_edges
            << ") for " << num_nodes << " nodes.\n";
        num_edges = max_edges;
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
        {1, 2}, {2, 1}, 
        {2, 3}, {3, 2}, 
        {3, 4}, {4, 3},
        {3, 6}, {6, 3},
        {2, 5}, {5, 2}, 
        {5, 1}, {1, 5}, 
        {4, 6}, {6, 4},
        {1, 7}, {7, 1},
        {7, 8}, {8, 7},
        {8, 9}, {9, 8},
        {9, 10}, {10, 9},
        {10, 4}, {4, 10},
        {5, 11}, {11, 5},
        {11, 12}, {12, 11},
        {12, 13}, {13, 12},
        {13, 6}, {6, 13},
    };

    std::vector<EdgeProperties> edge_properties = {
        {20}, {20},
        {10}, {10},
        {20}, {20},
        {20}, {20},
        {30}, {30},
        {600}, {600},
        {400}, {400},
        {100}, {100},
        {100}, {100},
        {100}, {100},
        {200}, {200},
        {600}, {600},
        {600}, {600},
        {600}, {600},
        {400}, {400},
        {400}, {400},
    };

    Graph g;

    for (size_t i = 0; i < edges.size(); ++i) {
        auto e = boost::add_edge(edges[i].first, edges[i].second, g).first;
        g[e].capacity = edge_properties[i].capacity;
		g[e].flow = edge_properties[i].flow;
		g[e].weight = 100 / g[e].capacity;
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

int main() {
    try {
        // Graph
        long long num_nodes, num_edges;

        // Commodity
        int num_commodities, min_demand, max_demand;       

        int num_threads, num_of_iter;

        /*cout << "Enter the number of nodes: ";
        cin >> num_nodes;
        cout << "Enter the number of edges: ";
        cin >> num_edges;
        cout << "Enter the number of commodities: ";
        cin >> num_commodities;
        cout << "Enter the minimum demand for each commodities: ";
        cin >> min_demand;
        cout << "Enter the maximum demand for each commodities: ";
        cin >> max_demand;*/
        cout << "Enter the number of iterations: ";
        cin >> num_of_iter;
        cout << "Enter the number of threads to use: ";
        cin >> num_threads;

        Graph g = graph_test_init();    
        //Graph g = generate_random_graph(num_nodes, num_edges);

        //std::vector<Commodity> commodities = generate_random_commodities(num_commodities, g, min_demand, max_demand);
	    std::vector<Commodity> commodities = {
		    {1, 4, 200},
		    {5, 6, 400},
	    };

        cout << "\n== Initial Commodities before Flow Distribution ==" << endl;
        for (const auto& commodity : commodities) {
            cout << "Commodity: Source = " << commodity.source
                << ", Destination = " << commodity.destination
                << ", Demand = " << commodity.demand << endl;
        }

        for (auto e : boost::make_iterator_range(boost::edges(g))) {
            auto source_node = boost::source(e, g);
            auto target_node = boost::target(e, g);

            auto flow = g[e].flow;
            auto capacity = g[e].capacity;

            
            std::cout << source_node << " -> " << target_node
                << " [Flow: " << flow << ", Capacity: " << capacity 
                << ", Weight: " << g[e].weight << "]\n";
            
        }

		/*int max_threads = omp_get_max_threads();
		cout << "Max threads: " << max_threads << endl;*/
        omp_set_num_threads(num_threads);
        double omp_start = omp_get_wtime();
        OMP_flowDistributionAlgorithm(g, commodities, num_of_iter);
        double omp_end = omp_get_wtime();

        //reset_flow_and_commodity(g, commodities);

        /*double ori_start = omp_get_wtime();
        flowDistributionAlgorithm(g, commodities, num_of_iter);
        double ori_end = omp_get_wtime();*/

        for (auto e : boost::make_iterator_range(boost::edges(g))) {
            auto source_node = boost::source(e, g);
            auto target_node = boost::target(e, g);

            auto flow = g[e].flow;
            auto capacity = g[e].capacity;

            if (g[e].flow > 0) {
                std::cout << source_node << " -> " << target_node
                    << " [Flow: " << flow << ", Capacity: " << capacity << "]\n";
            }
        }

        // write into file
        /*std::ofstream mainFile("..\\Python-TFS\\cuda_omp_st.txt");
        std::ofstream ompFile("..\\Python-TFS\\omp.txt", std::ios::app);
        std::ofstream stFile("..\\Python-TFS\\st.txt", std::ios::app);*/

        //if (!mainFile && !ompFile && !stFile) {
        //    std::cerr << "Error opening file!" << std::endl;
        //    return -1;
        //}

        // write some text to the file
        //mainFile << "ST, " << ori_runtime << std::endl;
        //mainFile << "OMP, " << omp_runtime << std::endl;

        //ompFile << "(" << boost::num_vertices(g) << ", " << commodities.size() << ", " << omp_runtime << ")" << endl;
        //stFile << "(" << boost::num_vertices(g) << ", " << commodities.size() << ", " << ori_runtime << ")" << endl;

        // close the file
        /*mainFile.close();
        ompFile.close();
        stFile.close();*/

		displayCommodityPaths(g, commodities);

        double omp_runtime = omp_end - omp_start;
        //double ori_runtime = ori_end - ori_start;

        //cout << "Original Runtime: " << ori_runtime << endl;
        cout << "OMP Runtime: " << omp_runtime << endl;

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