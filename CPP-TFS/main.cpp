#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "PropFlowAlgorithm.hpp"
#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    NetworkGraph graph;

    graph.addEdge("A", "B", 7, 16);
    graph.addEdge("A", "C", 4, 10);
    graph.addEdge("A", "D", 3, 9); 
    graph.addEdge("B", "D", 4, 9);  
    graph.addEdge("B", "E", 6, 13);  
    graph.addEdge("C", "D", 2, 6); 
    graph.addEdge("C", "F", 10, 20);
    graph.addEdge("D", "E", 7, 15);  
    graph.addEdge("E", "G", 2, 6); 
    graph.addEdge("E", "F", 4, 10);  
    graph.addEdge("F", "G", 3, 8); 
    //graph.addEdge("F", "I", 8, 20); 
    graph.addEdge("G", "B", 7, 18); 
    /*graph.addEdge("G", "H", 5, 12); 
    graph.addEdge("G", "I", 6, 14); 
    graph.addEdge("H", "B", 11, 22);
    graph.addEdge("H", "I", 5, 8); */

    // define commodities
    vector<pair<string, string>> commodities = {
        {"A", "G"},{"C", "F"},{"B", "G"},   
    };
    vector<double> demands = { 20, 15, 20 };  // demands for each commodity

    double omp_start = omp_get_wtime();
    OMP_equalDistributionAlgorithm(graph, commodities, demands);
    double omp_end = omp_get_wtime();

    double omp_rt = omp_end - omp_start;

    cout << "\nFinal Flows After Proportional Balancing:\n";
    for (const Edge& e : graph.getEdges()) {
        cout << "Edge " << e.source << " -> " << e.destination
            << " | Flow: " << e.flow << "/" << e.capacity << "\n";
    }

    cout << "\nOMP runtime: " << omp_rt << endl;

    return 0;
}
