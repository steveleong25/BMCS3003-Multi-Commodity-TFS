#include "NetworkGraph.hpp"

int main() {
    NetworkGraph graph;

    // Add edges to the graph
    graph.addEdge("A", "B", 7, 16, true);
    graph.addEdge("A", "C", 4, 10, false);
    graph.addEdge("A", "D", 3, 8, true); 
    graph.addEdge("B", "D", 4, 10, false);  
    graph.addEdge("B", "E", 6, 14, false);  
    graph.addEdge("C", "D", 2, 6, false); 
    graph.addEdge("D", "E", 7, 16, true);  
    graph.addEdge("E", "G", 2, 6, true); 
    graph.addEdge("E", "F", 5, 12, true);  
    graph.addEdge("F", "G", 3, 8, false); 
    graph.addEdge("G", "B", 10, 22, false); 

    // Display the graph
    graph.displayGraph();

    return 0;
}
