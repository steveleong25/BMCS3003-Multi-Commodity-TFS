#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "PropFlowAlgorithm.hpp"
#include <vector>
#include <string>
#include <iostream>

using namespace std;

int main() {
    NetworkGraph graph;

    graph.addEdge("A", "B", 7, 16);
    graph.addEdge("A", "C", 4, 10);
    graph.addEdge("A", "D", 3, 9); 
    graph.addEdge("B", "D", 4, 9);  
    graph.addEdge("B", "E", 6, 13);  
    graph.addEdge("C", "D", 2, 6); 
    graph.addEdge("D", "E", 7, 15);  
    graph.addEdge("E", "G", 2, 6); 
    graph.addEdge("E", "F", 5, 11);  
    graph.addEdge("F", "G", 3, 8); 
    graph.addEdge("G", "B", 7, 18); 

    // Step 2: Define commodities (source-destination pairs) and their demands
    vector<pair<string, string>> commodities = {
        {"A", "G"},   // Commodity 1: From A to G
        {"C", "F"},   // Commodity 2: From A to F
        {"B", "G"}    // Commodity 3: From B to G
    };
    vector<double> demands = { 20, 15, 10 };  // Demands for each commodity

    //cout << "Sending " << demands[0] << ", " << demands[1] << ", " << demands[2] << "...\n";

    // Step 3: Run the proportional congestion balancing algorithm
    equalDistributionAlgorithm(graph, commodities, demands);

    // Step 4: Display final results
    cout << "\nFinal Flows After Proportional Balancing:\n";
    for (const Edge& e : graph.getEdges()) {
        cout << "Edge " << e.source << " -> " << e.destination
            << " | Flow: " << e.flow << "/" << e.capacity << "\n";
    }

    return 0;
}
