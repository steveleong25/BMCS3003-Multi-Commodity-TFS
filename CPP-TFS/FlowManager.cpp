#include "FlowManager.hpp"
#include <fstream>

void FlowManager::addFlow(std::vector<Edge>& edges, const std::string& src, const std::string& dest, int flowToAdd) {
    for (auto& edge : edges) {
        if (edge.source == src && edge.destination == dest) {
            if (edge.flow + flowToAdd <= edge.capacity) {
                edge.flow += flowToAdd;
                std::cout << "Flow added: " << flowToAdd << " from " << src << " to " << dest << std::endl;
            }
            else {
                std::cout << "Cannot add flow: Exceeds capacity of " << edge.capacity << std::endl;
            }
            break;
        }
    }
}

void FlowManager::removeFlow(std::vector<Edge>& edges, const std::string& src, const std::string& dest, int flowToRemove) {
    for (auto& edge : edges) {
        if (edge.source == src && edge.destination == dest) {
            if (edge.flow - flowToRemove >= 0) {
                edge.flow -= flowToRemove;
                std::cout << "Flow removed: " << flowToRemove << " from " << src << " to " << dest << std::endl;
            }
            else {
                std::cout << "Cannot remove flow: Insufficient flow on this edge" << std::endl;
            }
            break;
        }
    }
}

void FlowManager::simulateFlowOverTime(std::vector<Edge>& edges, const std::vector<Commodity>& commodities, int timeSteps, const std::string& outputFile) {
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Could not open output file for writing.\n";
        return;
    }

    // Header for the output file (for visualization)
    outFile << "Time,Source,Destination,Flow\n";

    // Iterate through timesteps
    for (int t = 0; t < timeSteps; ++t) {
        std::cout << "Time step: " << t << std::endl;

        // Process each commodity
        for (const auto& commodity : commodities) {
            // Find a multi-hop path from source to destination
            auto path = findShortestPath(edges, commodity.source, commodity.destination);

            if (path.empty()) {
                std::cout << "No path found for commodity: " << commodity.source << " -> " << commodity.destination << "\n";
                continue;
            }

            int flowToAdd = commodity.demand; // Assume full demand is added

            // Update flow along the path
            for (size_t i = 0; i < path.size() - 1; ++i) {
                const std::string& src = path[i];
                const std::string& dest = path[i + 1];

                for (auto& edge : edges) {
                    if (edge.source == src && edge.destination == dest) {
                        if (edge.flow + flowToAdd <= edge.capacity) {
                            edge.flow += flowToAdd;
                            std::cout << "Added flow: " << flowToAdd << " from " << src << " to " << dest << "\n";
                            outFile << t << "," << src << "," << dest << "," << edge.flow << "\n";
                        }
                        else {
                            std::cout << "Cannot add flow: Exceeds capacity on edge " << src << " -> " << dest << "\n";
                        }
                        break;
                    }
                }
            }
        }

        // Remove some flow from edges to simulate vehicles leaving
        for (auto& edge : edges) {
            if (edge.flow > 0) {
                edge.flow = std::max(0, edge.flow - 1); // Simulate flow leaving
            }
        }
    }

    outFile.close();
}

