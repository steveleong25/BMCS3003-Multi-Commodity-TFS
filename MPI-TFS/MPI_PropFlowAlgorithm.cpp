#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "NetworkGraph.hpp"
#include "PropFlowAlgorithm.hpp"
#include "PathFinder.hpp"
#include <sstream>

using namespace std;

const double INF = numeric_limits<double>::max();


void MPI_redistributeFlowForEqualization(NetworkGraph& graph,
    vector<pair<string, string>>& commodities,
    vector<double>& demands,
    vector<double>& unitsDelivered,
    vector<double>& successRates,
    int rank, int size) {

    double totalUnitsDelivered = 0;
    double totalDemand = 0;

    // Calculate total units delivered and total demand using MPI
	if (rank == 0) {
		for (int i = 0; i < commodities.size(); i++) {
			totalUnitsDelivered += unitsDelivered[i];
			totalDemand += demands[i];
		}
	}

    double equalSuccessRate = totalUnitsDelivered / totalDemand;
    vector<double> unusedFlow(commodities.size(), 0.0);

    // Redistribution of excess flow
    if (rank == 0)
    {
        for (int i = 0; i < commodities.size(); i++) {
            if (successRates[i] > equalSuccessRate) {
                double excessFlow = floor((successRates[i] - equalSuccessRate) * demands[i]);
                const string& source = commodities[i].first;
                const string& destination = commodities[i].second;

                vector<vector<string>> paths = findAllPaths(graph.getEdges(), source, destination);

                for (const auto& path : paths) {
                    if (excessFlow <= 0) break;

                    double pathFlow = INF;

                    // calculate min flow
                    for (int j = 0; j < path.size() - 1; ++j) {
                        Edge& edge = graph.getEdge(path[j], path[j + 1]);
                        pathFlow = min(pathFlow, static_cast<double>(edge.flow));
                    }

                    double flowToRemove = min(excessFlow, pathFlow);
                    if (flowToRemove > 0) {
                        sendFlow(graph, path, -flowToRemove);
                        unitsDelivered[i] -= flowToRemove;
                        excessFlow -= flowToRemove;
                        unusedFlow[i] += flowToRemove;
                    }
                }
            }
        }
    }

    // Redistribution of flow
    if (rank == 0)
    {
        for (int i = 0; i < commodities.size(); i++) {
            if (successRates[i] < equalSuccessRate) {
                double neededFlow = ceil((equalSuccessRate - successRates[i]) * demands[i]);
                const string& source = commodities[i].first;
                const string& destination = commodities[i].second;

                vector<vector<string>> paths = findAllPaths(graph.getEdges(), source, destination);

                for (const auto& path : paths) {
                    if (neededFlow <= 0) break;

                    double pathCapacity = INF;

                    // calculate available capacity in the path
                    for (int j = 0; j < path.size() - 1; ++j) {
                        Edge& edge = graph.getEdge(path[j], path[j + 1]);
                        pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
                    }

                    double flowToAdd = min(neededFlow, pathCapacity);
                    if (flowToAdd > 0) {
                        sendFlow(graph, path, flowToAdd);
                        unitsDelivered[i] += flowToAdd;
                        neededFlow -= flowToAdd;
                        unusedFlow[i] -= flowToAdd;
                    }
                }
            }
        }

        // Redistribute flow post-redistribution (for unused flow)
        for (int i = 0; i < commodities.size(); i++) {
            if (unusedFlow[i] > 0) {
                const string& source = commodities[i].first;
                const string& destination = commodities[i].second;

                vector<vector<string>> paths = findAllPaths(graph.getEdges(), source, destination);

                for (const auto& path : paths) {
                    if (unusedFlow[i] <= 0) break;

                    double pathCapacity = INF;

                    // Calculate the available capacity in the path
                    for (int j = 0; j < path.size() - 1; ++j) {
                        Edge& edge = graph.getEdge(path[j], path[j + 1]);
                        pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
                    }

                    double flowToAdd = min(unusedFlow[i], pathCapacity);
                    if (flowToAdd > 0) {
                        sendFlow(graph, path, flowToAdd);
                        unitsDelivered[i] += flowToAdd;
                        unusedFlow[i] -= flowToAdd;
                    }
                }
            }
        }
    }
}


void MPI_equalDistributionAlgorithm(NetworkGraph& graph,
    vector<pair<string, string>> commodities,
    vector<double> demands,
    int rank, int size) {

    vector<double> unitsDelivered(commodities.size(), 0.0);
    vector<double> successRates(commodities.size(), 0.0);

    bool moreFlowNeeded = true;

    while (moreFlowNeeded) {
        moreFlowNeeded = false;

        // Update success rates and other local work
        if (rank == 0) {
            for (int i = 0; i < commodities.size(); i++) {
                if (demands[i] > 0) {
                    successRates[i] = unitsDelivered[i] / demands[i];
                }
                else {
                    successRates[i] = 1.0;
                }
            }
        }

        int lowestIndex = 0;
        if (rank == 0) {
            for (int i = 1; i < successRates.size(); i++) {
                if (successRates[i] < successRates[lowestIndex]) {
                    lowestIndex = i;
                }
            }
        }
		
        const string& source = commodities[lowestIndex].first;
        const string& destination = commodities[lowestIndex].second;
        double remainingDemand = demands[lowestIndex] - unitsDelivered[lowestIndex];

        vector<vector<string>> allPaths = findAllPaths(graph.getEdges(), source, destination);

        // Flow allocation in parallel
        if (rank == 0)
        {
            for (int i = 0; i < allPaths.size(); i++) {
				//std::cout << "Remaining Demand: " << remainingDemand << std::endl;
                if (remainingDemand <= 0) continue;
                double pathCapacity = INF;

                // Calculate bottleneck capacity
                for (int j = 0; j < allPaths[i].size() - 1; ++j) {
                    Edge& edge = graph.getEdge(allPaths[i][j], allPaths[i][j + 1]);
                    pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
                }

                if (pathCapacity > 0) {
                    moreFlowNeeded = true;

                    double flowToSend = min(remainingDemand, pathCapacity);
                    sendFlow(graph, allPaths[i], flowToSend);

                    unitsDelivered[lowestIndex] += flowToSend;
                    remainingDemand -= flowToSend;
                }
            }
        }
    }

    // Print final results
    if (rank == 0) {
        std::cout << "\nFinal Results: Units Successfully Reaching Destinations (Before Redistribution)\n";
        for (int i = 0; i < commodities.size(); ++i) {
            std::cout << "Commodity " << i + 1 << " (From " << commodities[i].first
                << " to " << commodities[i].second << "): "
                << unitsDelivered[i] << "/" << demands[i] << " units\n";
        }


        MPI_redistributeFlowForEqualization(graph, commodities, demands, unitsDelivered, successRates, rank, size);
    
        std::cout << "\nFinal Results: Units Successfully Reaching Destinations (After Redistribution)\n";
        for (int i = 0; i < commodities.size(); ++i) {
            std::cout << "Commodity " << i + 1 << " (From " << commodities[i].first
                << " to " << commodities[i].second << "): "
                << unitsDelivered[i] << "/" << demands[i] << " units\n";
        }
    }

}
