#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "NetworkGraph.hpp"
#include "PropFlowAlgorithm.hpp"
#include "PathFinder.hpp"

using namespace std;

const double INF = numeric_limits<double>::max();

void MPI_redistributeFlowForEqualization(NetworkGraph& graph,
    vector<pair<string, string>>& commodities,
    vector<double>& demands,
    vector<double>& unitsDelivered,
    vector<double>& successRates) {

    double totalUnitsDelivered = 0;
    double totalDemand = 0;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate total units delivered and total demand using MPI
    for (int i = 0; i < commodities.size(); ++i) {
        totalUnitsDelivered += unitsDelivered[i];
        totalDemand += demands[i];
    }
    

    // Broadcast totalUnitsDelivered and totalDemand to all processes
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&totalUnitsDelivered, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&totalDemand, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double equalSuccessRate = totalUnitsDelivered / totalDemand;
    vector<double> unusedFlow(commodities.size(), 0.0);

    // Redistribution of excess flow
    for (int i = rank; i < commodities.size(); i += size) {
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

    // Gather unusedFlow from all processes
    vector<double> gatheredUnusedFlow(commodities.size(), 0.0);
    MPI_Reduce(&unusedFlow[rank], &gatheredUnusedFlow[0], unusedFlow.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&gatheredUnusedFlow, gatheredUnusedFlow.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Update unusedFlow with gathered data
    for (int i = 0; i < commodities.size(); ++i) {
        unusedFlow[i] = gatheredUnusedFlow[i];
    }

    // Redistribution of flow
    for (int i = rank; i < commodities.size(); i += size) {
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
    for (int i = rank; i < commodities.size(); i += size) {
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

    // Gather final unitsDelivered from all processes
    vector<double> gatheredUnitsDelivered(commodities.size(), 0.0);
    MPI_Reduce(&unitsDelivered[rank], &gatheredUnitsDelivered[0], unitsDelivered.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&gatheredUnitsDelivered, gatheredUnitsDelivered.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Update unitsDelivered with gathered data
    for (int i = 0; i < commodities.size(); ++i) {
        unitsDelivered[i] = gatheredUnitsDelivered[i];
    }
}


void MPI_equalDistributionAlgorithm(NetworkGraph& graph,
    vector<pair<string, string>> commodities,
    vector<double> demands) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<double> unitsDelivered(commodities.size(), 0.0);
    vector<double> successRates(commodities.size(), 0.0);

    bool moreFlowNeeded = true;

    while (true) {
        bool localMoreFlowNeeded = false;

        // Update success rates and other local work
        for (int i = rank; i < commodities.size(); i += size) {
            if (demands[i] > 0) {
                successRates[i] = unitsDelivered[i] / demands[i];
            }
            else {
                successRates[i] = 1.0;
            }
        }

        // Find the lowest success rate in parallel
        int lowestIndex = 0;
        for (int i = 1; i < successRates.size(); ++i) {
            if (successRates[i] < successRates[lowestIndex]) {
                lowestIndex = i;
            }
        }

        const string& source = commodities[lowestIndex].first;
        const string& destination = commodities[lowestIndex].second;
        double remainingDemand = demands[lowestIndex] - unitsDelivered[lowestIndex];

        vector<vector<string>> allPaths = findAllPaths(graph.getEdges(), source, destination);

        // Flow allocation in parallel
        for (int i = rank; i < allPaths.size(); i += size) {
            if (remainingDemand <= 0) continue;

            double pathCapacity = INF;

            // Calculate bottleneck capacity
            for (int j = 0; j < allPaths[i].size() - 1; ++j) {
                Edge& edge = graph.getEdge(allPaths[i][j], allPaths[i][j + 1]);
                pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
            }

            if (pathCapacity > 0) {
                localMoreFlowNeeded = true;  // Set local flag

                double flowToSend = min(remainingDemand, pathCapacity);
                sendFlow(graph, allPaths[i], flowToSend);

                unitsDelivered[lowestIndex] += flowToSend;
                remainingDemand -= flowToSend;
            }
        }

        // Gather updated unitsDelivered
        vector<double> gatheredUnitsDelivered(commodities.size(), 0.0);
        MPI_Reduce(&unitsDelivered[rank], &gatheredUnitsDelivered[0], unitsDelivered.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&gatheredUnitsDelivered, gatheredUnitsDelivered.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Update unitsDelivered with gathered data
        for (int i = 0; i < commodities.size(); ++i) {
            unitsDelivered[i] = gatheredUnitsDelivered[i];
        }

        // Perform global reduction to determine if any process needs more flow
        bool globalMoreFlowNeeded = false;
        MPI_Allreduce(&localMoreFlowNeeded, &globalMoreFlowNeeded, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        // Break the loop if no more flow is needed
        if (!globalMoreFlowNeeded) break;
    }

    // Print final results
    if (rank == 0) {
        std::cout << "\nFinal Results: Units Successfully Reaching Destinations (Before Redistribution)\n";
        for (int i = 0; i < commodities.size(); ++i) {
            std::cout << "Commodity " << i + 1 << " (From " << commodities[i].first
                << " to " << commodities[i].second << "): "
                << unitsDelivered[i] << "/" << demands[i] << " units\n";
        }

        // Call redistribute flow for equalization
        MPI_redistributeFlowForEqualization(graph, commodities, demands, unitsDelivered, successRates);

        std::cout << "\nFinal Results: Units Successfully Reaching Destinations (After Redistribution)\n";
        for (int i = 0; i < commodities.size(); ++i) {
            std::cout << "Commodity " << i + 1 << " (From " << commodities[i].first
                << " to " << commodities[i].second << "): "
                << unitsDelivered[i] << "/" << demands[i] << " units\n";
        }
    }
}