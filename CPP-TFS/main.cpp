#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "PropFlowAlgorithm.hpp"
#include "Commodity.hpp"
//#include "CUDAFlowAlgorithm.hpp"
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/random.hpp>
#include <omp.h>

using namespace std;
//extern "C" void CUDA_equalDistributionAlgorithm(NetworkGraph & graph, const std::vector<std::pair<std::string, std::string>>&commodities, const std::vector<double>&demands);


//int main() {
//    NetworkGraph graph;
//
//    graph.addEdge("A", "B", 7, 16);
//    graph.addEdge("A", "C", 4, 10);
//    graph.addEdge("A", "D", 3, 9); 
//    graph.addEdge("B", "D", 4, 9);  
//    graph.addEdge("B", "E", 6, 13);  
//    graph.addEdge("C", "D", 2, 6); 
//    graph.addEdge("C", "F", 10, 20);
//    graph.addEdge("D", "E", 7, 15);  
//    graph.addEdge("E", "G", 2, 6); 
//    graph.addEdge("E", "F", 4, 10);  
//    graph.addEdge("F", "G", 3, 8); 
//    //graph.addEdge("F", "I", 8, 20); 
//    graph.addEdge("G", "B", 7, 18); 
//    /*graph.addEdge("G", "H", 5, 12); 
//    graph.addEdge("G", "I", 6, 14); 
//    graph.addEdge("H", "B", 11, 22);
//    graph.addEdge("H", "I", 5, 8); */
//
//    // define commodities
//    vector<pair<string, string>> commodities = {
//        {"A", "G"},{"C", "F"},{"B", "G"},   
//    };
//    vector<double> demands = { 20, 15, 20 };  // demands for each commodity
//
//    double ori_start = omp_get_wtime();
//    equalDistributionAlgorithm(graph, commodities, demands);
//    double ori_end = omp_get_wtime();
//    
//	graph.resetFlow();
//
//    double omp_start = omp_get_wtime();
//    //equalDistributionAlgorithm(graph, commodities, demands);
//    OMP_equalDistributionAlgorithm(graph, commodities, demands);
//    //CUDA_equalDistributionAlgorithm(graph, commodities, demands);
//
//    double omp_end = omp_get_wtime();
//
//	double ori_rt = ori_end - ori_start;
//    double omp_rt = omp_end - omp_start;
//
//    cout << "\nFinal Flows After Proportional Balancing:\n";
//    for (const Edge& e : graph.getEdges()) {
//        cout << "Edge " << e.source << " -> " << e.destination
//            << " | Flow: " << e.flow << "/" << e.capacity << "\n";
//    }
//
//    cout << "\nOriginal runtime: " << ori_rt << endl;
//    cout << "\nOMP runtime: " << omp_rt << endl;
//
//
//    return 0;
//}

using namespace boost;

// Define the graph type
Graph generate_random_graph(int num_nodes, int num_edges) {
    Graph g(num_nodes);
    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist_weight(1, 10);  // Weight range [1, 10]
    boost::random::uniform_int_distribution<> dist_capacity(10, 50); // Capacity range [10, 50]

    // Generate random edges
    for (int i = 0; i < num_edges; ++i) {
        int u = gen() % num_nodes; //source
		int v = gen() % num_nodes; //destination

        if (u != v) {
            auto e = boost::add_edge(u, v, g).first;
            g[e].capacity = dist_capacity(gen);  // Set the capacity for the edge
            g[e].flow = 0;  // Initialize the flow to 0

            put(boost::edge_weight, g, e, dist_weight(gen));
        }
    }

    std::cout << "Graph with " << num_nodes << " nodes and " << num_edges << " edges created." << std::endl;

    std::cout << "Nodes in the graph: ";
    for (auto v : boost::make_iterator_range(boost::vertices(g))) {
        std::cout << v << " ";
    }
    std::cout << "\n";

    // Verify edges
    std::cout << "Edges in the graph:\n";
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        auto source_node = boost::source(e, g);
        auto target_node = boost::target(e, g);

        // Get edge properties
        auto weight = get(boost::edge_weight, g, e);
        auto capacity = g[e].capacity;

        std::cout << source_node << " -> " << target_node
            << " [Weight: " << weight << ", Capacity: " << capacity << "]\n";
    }

    return g;
}

Graph graph_test_init() {
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 0}, // Bidirectional edge between 0 and 1
        {1, 2}, {2, 1}, // Bidirectional edge between 1 and 2
        {2, 3}, {3, 2}, // Bidirectional edge between 2 and 3
        {2, 5}, {5, 2}, // Bidirectional edge between 2 and 5
        {4, 1}, {1, 4}  // Bidirectional edge between 4 and 1
    };

    std::vector<EdgeProperties> edge_properties = {
        {10, 10}, {10, 10}, // Weight and capacity for edge 0 <-> 1
        {12, 20}, {12, 20}, // Weight and capacity for edge 1 <-> 2
        {8, 15}, {8, 15},   // Weight and capacity for edge 2 <-> 3
        {10, 40}, {10, 40}, // Weight and capacity for edge 2 <-> 5
        {15, 30}, {15, 30}  // Weight and capacity for edge 4 <-> 1
    };

    Graph g;

    for (size_t i = 0; i < edges.size(); ++i) {
        auto e = boost::add_edge(edges[i].first, edges[i].second, g).first;
        g[e].capacity = edge_properties[i].capacity;
		g[e].flow = edge_properties[i].flow;
		put(boost::edge_weight, g, e, edge_properties[i].weight);
    }

    /*auto e = boost::add_edge(0, 1, g).first;
    g[e].capacity = 20;
    g[e].flow = 0;  
    put(boost::edge_weight, g, e, 10);
    
    auto e1 = boost::add_edge(1, 2, g).first;
    g[e1].capacity = 20;
    g[e1].flow = 0;  
    put(boost::edge_weight, g, e1, 12);
    
    auto e2 = boost::add_edge(2, 3, g).first;
    g[e2].capacity = 15;
    g[e2].flow = 0;  
    put(boost::edge_weight, g, e2, 8);
    
    auto e3 = boost::add_edge(4, 1, g).first;
    g[e3].capacity = 25;
    g[e3].flow = 0;  
    put(boost::edge_weight, g, e3, 15);
    
    auto e4 = boost::add_edge(2, 5, g).first;
    g[e4].capacity = 10;
    g[e4].flow = 0;  
    put(boost::edge_weight, g, e4, 10);*/

    std::cout << "Edges in the graph:\n";
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        auto source_node = boost::source(e, g);
        auto target_node = boost::target(e, g);

        // Get edge properties
        auto weight = get(boost::edge_weight, g, e);
        auto capacity = g[e].capacity;

        std::cout << source_node << " -> " << target_node
            << " [Weight: " << weight << ", Capacity: " << capacity << "]\n";
    }

    return g;
}

int main() 
{
    //Graph
    //int num_nodes = 6; // Adjust nodes
    //int num_edges = 24; // Desired number of edges
    //Graph g = generate_random_graph(num_nodes, num_edges);
    Graph g = graph_test_init();    

    //Commodity
    int num_commodities = 3;  // number of commodities
    //int min_demand = 10;      // Minimum demand for a commodity
    //int max_demand = 100;     // Maximum demand for a commodity

    graph_traits<Graph>::vertex_iterator vi, vi_end;
    tie(vi, vi_end) = boost::vertices(g);

    //std::vector<Commodity> commodities = generate_random_commodities(num_commodities, g);
	std::vector<Commodity> commodities = {
		{0, 3, 2},
		{4, 5, 5},
	};
    double omp_start = omp_get_wtime();
    for (const auto& commodity : commodities) {
        std::cout << "Commodity: Source = " << commodity.source
            << ", Destination = " << commodity.destination
            << ", Demand = " << commodity.demand << std::endl;
    }

	std::vector<std::vector<int>> shortest_paths;
    for (int i = 0; i < commodities.size(); i++) {
		shortest_paths.push_back(find_shortest_path(g, commodities[i].source, commodities[i].destination));
    }

    if (!shortest_paths.empty()) {
        for (size_t i = 0; i < shortest_paths.size(); ++i) {
            std::cout << "Shortest path from " << commodities[i].source << " to " << commodities[i].destination << ":\n";

            if (shortest_paths[i].empty()) {
                std::cout << "No valid path found.\n";
            }
            else {
                for (size_t j = 0; j < shortest_paths[i].size(); ++j) {
                    std::cout << shortest_paths[i][j];
                    if (j < shortest_paths[i].size() - 1) {
                        std::cout << " -> ";
                    }
                }
                std::cout << "\n";
            }
        }
    }

	double ratio = OMP_flowDistributionAlgorithm(g, commodities, 0.01, 0.1);
    
    double omp_end = omp_get_wtime();
    double omp_runtime = omp_end - omp_start;

	cout << "Max ratio: " << ratio << endl;
    cout << "Runtime: " << omp_runtime << endl;
    return 0;
}