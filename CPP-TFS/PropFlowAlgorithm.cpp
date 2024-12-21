#include <iostream>
#include <mutex>
#include <omp.h>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"

using namespace std;

std::mutex flowMutex;

const double INF = numeric_limits<double>::max();

void sendFlow(NetworkGraph& graph, const vector<string>& path, double amount) {
    for (int i = 0; i < path.size() - 1; ++i) {
        Edge& edge = graph.getEdge(path[i], path[i + 1]);
        edge.flow += amount;
    }
}

void redistributeFlowForEqualization(NetworkGraph& graph,
    vector<pair<string, string>>& commodities,
    vector<double>& demands,
    vector<double>& unitsDelivered,
    vector<double>& successRates) {

    double totalUnitsDelivered = 0;
    double totalDemand = 0;
    for (size_t i = 0; i < commodities.size(); ++i) {
        totalUnitsDelivered += unitsDelivered[i];
        totalDemand += demands[i];
    }
    double equalSuccessRate = totalUnitsDelivered / totalDemand;

    vector<double> unusedFlow(commodities.size(), 0.0);

    // redistribution of flow
    for (size_t i = 0; i < commodities.size(); ++i) {
        if (successRates[i] > equalSuccessRate) {
            double excessFlow = floor((successRates[i] - equalSuccessRate) * demands[i]);
            const string& source = commodities[i].first;
            const string& destination = commodities[i].second;

            vector<vector<string>> paths = findAllPaths(graph.getEdges(), source, destination);
            for (const auto& path : paths) {
                if (excessFlow <= 0) break;
                double pathFlow = INF;
                for (size_t j = 0; j < path.size() - 1; ++j) {
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

    for (size_t i = 0; i < commodities.size(); ++i) {
        if (successRates[i] < equalSuccessRate) {
            double neededFlow = ceil((equalSuccessRate - successRates[i]) * demands[i]);
            const string& source = commodities[i].first;
            const string& destination = commodities[i].second;

            vector<vector<string>> paths = findAllPaths(graph.getEdges(), source, destination);
            for (const auto& path : paths) {
                if (neededFlow <= 0) break;

                double pathCapacity = INF;
                for (size_t j = 0; j < path.size() - 1; ++j) {
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

    // post-redistribution adjustment to reallocate unused flow
    for (size_t i = 0; i < commodities.size(); ++i) {
        if (unusedFlow[i] > 0) {
            const string& source = commodities[i].first;
            const string& destination = commodities[i].second;

            vector<vector<string>> paths = findAllPaths(graph.getEdges(), source, destination);
            for (const auto& path : paths) {
                if (unusedFlow[i] <= 0) break;

                double pathCapacity = INF;
                for (size_t j = 0; j < path.size() - 1; ++j) {
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

void equalDistributionAlgorithm(NetworkGraph& graph,
    vector<pair<string, string>> commodities,
    vector<double> demands) {
    vector<double> unitsDelivered(commodities.size(), 0.0); 
    vector<double> successRates(commodities.size(), 0.0); 

    bool moreFlowNeeded = true;

    while (moreFlowNeeded) {
        moreFlowNeeded = false;

        // update success rates
        for (size_t i = 0; i < commodities.size(); ++i) {
            if (demands[i] > 0) {
                successRates[i] = unitsDelivered[i] / demands[i];
            }
            else {
                successRates[i] = 1.0;
            }
        }

        // find commodity with the lowest success rate
        size_t lowestIndex = 0;
        for (size_t i = 1; i < successRates.size(); ++i) {
            if (successRates[i] < successRates[lowestIndex]) {
                lowestIndex = i;
            }
        }

        const string& source = commodities[lowestIndex].first;
        const string& destination = commodities[lowestIndex].second;
        double remainingDemand = demands[lowestIndex] - unitsDelivered[lowestIndex];

        // get all paths from source to dest
        vector<vector<string>> allPaths = findAllPaths(graph.getEdges(), source, destination);

        for (const auto& path : allPaths) {
            if (remainingDemand <= 0) break;

            // calculate bottleneck capacity
            double pathCapacity = INF;
            for (size_t j = 0; j < path.size() - 1; ++j) {
                Edge& edge = graph.getEdge(path[j], path[j + 1]);
                pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
            }

            if (pathCapacity > 0) {
                moreFlowNeeded = true;

                // allocate flow
                double flowToSend = min(remainingDemand, pathCapacity);
                for (int i = 0; i < path.size(); ++i) {
                    cout << "Path " << i + 1 << ": ";
                    for (int j = 0; j < path[i].size(); ++j) {
                        cout << path[i][j];
                        if (j < path[i].size() - 1) {
                            cout << " -> "; // Add an arrow between nodes
                        }
                    }
                    cout << endl; // Move to the next path
                }
                sendFlow(graph, path, flowToSend);

                // update demand and track delivered flow
                unitsDelivered[lowestIndex] += flowToSend;
                remainingDemand -= flowToSend;
            }
        }
    }

    cout << "\nFinal Results: Units Successfully Reaching Destinations (Before Redistribution)\n";
    for (size_t i = 0; i < commodities.size(); ++i) {
        cout << "Commodity " << i + 1 << " (From " << commodities[i].first
            << " to " << commodities[i].second << "): "
            << unitsDelivered[i] << "/" << demands[i] << " units\n";
    }

    redistributeFlowForEqualization(graph, commodities, demands, unitsDelivered, successRates);

    cout << "\nFinal Results: Units Successfully Reaching Destinations (After Redistribution)\n";
    for (size_t i = 0; i < commodities.size(); ++i) {
        cout << "Commodity " << i + 1 << " (From " << commodities[i].first
            << " to " << commodities[i].second << "): "
            << unitsDelivered[i] << "/" << demands[i] << " units\n";
    }


}

void OMP_redistributeFlowForEqualization(NetworkGraph& graph,
    vector<pair<string, string>>& commodities,
    vector<double>& demands,
    vector<double>& unitsDelivered,
    vector<double>& successRates) {

    double totalUnitsDelivered = 0;
    double totalDemand = 0;
    for (int i = 0; i < commodities.size(); ++i) {
        totalUnitsDelivered += unitsDelivered[i];
        totalDemand += demands[i];
    }
    double equalSuccessRate = totalUnitsDelivered / totalDemand;

    vector<double> unusedFlow(commodities.size(), 0.0);

    // Redistribution of flow in parallel
#pragma omp parallel for
    for (int i = 0; i < commodities.size(); ++i) {
        if (successRates[i] > equalSuccessRate) {
            double excessFlow = floor((successRates[i] - equalSuccessRate) * demands[i]);
            const string& source = commodities[i].first;
            const string& destination = commodities[i].second;

            vector<vector<string>> paths = findAllPaths(graph.getEdges(), source, destination);
            for (const auto& path : paths) {
                if (excessFlow <= 0) break;
                double pathFlow = INF;
                for (int j = 0; j < path.size() - 1; ++j) {
                    Edge& edge = graph.getEdge(path[j], path[j + 1]);
                    pathFlow = min(pathFlow, static_cast<double>(edge.flow));
                }

                double flowToRemove = min(excessFlow, pathFlow);
                if (flowToRemove > 0) {
                    // Lock the edge while modifying the flow
                    for (int j = 0; j < path.size() - 1; ++j) {
                        std::lock_guard<std::mutex> guard(flowMutex);  // Critical section for flow modification
                        Edge& edge = graph.getEdge(path[j], path[j + 1]);
                        sendFlow(graph, path, -flowToRemove);
                    }
                    unitsDelivered[i] -= flowToRemove;
                    excessFlow -= flowToRemove;
                    unusedFlow[i] += flowToRemove;
                }
            }
        }
    }

    // Adjustments for commodities with less flow
#pragma omp parallel for
    for (int i = 0; i < commodities.size(); ++i) {
        if (successRates[i] < equalSuccessRate) {
            double neededFlow = ceil((equalSuccessRate - successRates[i]) * demands[i]);
            const string& source = commodities[i].first;
            const string& destination = commodities[i].second;

            vector<vector<string>> paths = findAllPaths(graph.getEdges(), source, destination);
            for (const auto& path : paths) {
                if (neededFlow <= 0) break;

                double pathCapacity = INF;
                for (int j = 0; j < path.size() - 1; ++j) {
                    Edge& edge = graph.getEdge(path[j], path[j + 1]);
                    pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
                }

                double flowToAdd = min(neededFlow, pathCapacity);
                if (flowToAdd > 0) {
                    // Lock the edge while modifying the flow
                    for (int j = 0; j < path.size() - 1; ++j) {
                        std::lock_guard<std::mutex> guard(flowMutex);  // Critical section for flow modification
                        Edge& edge = graph.getEdge(path[j], path[j + 1]);
                        sendFlow(graph, path, flowToAdd);
                    }
                    unitsDelivered[i] += flowToAdd;
                    neededFlow -= flowToAdd;
                    unusedFlow[i] -= flowToAdd;
                }
            }
        }
    }

    // Post-redistribution adjustment for unused flow
#pragma omp parallel for
    for (int i = 0; i < commodities.size(); ++i) {
        if (unusedFlow[i] > 0) {
            const string& source = commodities[i].first;
            const string& destination = commodities[i].second;

            vector<vector<string>> paths = findAllPaths(graph.getEdges(), source, destination);
            for (const auto& path : paths) {
                if (unusedFlow[i] <= 0) break;

                double pathCapacity = INF;
                for (int j = 0; j < path.size() - 1; ++j) {
                    Edge& edge = graph.getEdge(path[j], path[j + 1]);
                    pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
                }

                double flowToAdd = min(unusedFlow[i], pathCapacity);
                if (flowToAdd > 0) {
                    // Lock the edge while modifying the flow
                    for (int j = 0; j < path.size() - 1; ++j) {
                        std::lock_guard<std::mutex> guard(flowMutex);  // Critical section for flow modification
                        Edge& edge = graph.getEdge(path[j], path[j + 1]);
                        sendFlow(graph, path, flowToAdd);
                    }
                    unitsDelivered[i] += flowToAdd;
                    unusedFlow[i] -= flowToAdd;
                }
            }
        }
    }
}

void OMP_equalDistributionAlgorithm(NetworkGraph& graph,
    vector<pair<string, string>> commodities,
    vector<double> demands) {

    vector<double> unitsDelivered(commodities.size(), 0.0);
    vector<double> successRates(commodities.size(), 0.0);

    bool moreFlowNeeded = true;

    while (moreFlowNeeded) {
        moreFlowNeeded = false;

        // Update success rates in parallel
        #pragma omp parallel for
        for (int i = 0; i < commodities.size(); ++i) {
            if (demands[i] > 0) {
                successRates[i] = unitsDelivered[i] / demands[i];
            }
            else {
                successRates[i] = 1.0;
            }
        }

        // Find commodity with the lowest success rate in parallel
        int lowestIndex = 0;
        #pragma omp parallel for
        for (int i = 1; i < successRates.size(); ++i) {
        #pragma omp critical
            {
                if (successRates[i] < successRates[lowestIndex]) {
                    lowestIndex = i;
                }
            }
        }

        const string& source = commodities[lowestIndex].first;
        const string& destination = commodities[lowestIndex].second;
        double remainingDemand = demands[lowestIndex] - unitsDelivered[lowestIndex];

        // Get all paths from source to destination
        vector<vector<string>> allPaths = findAllPaths(graph.getEdges(), source, destination);

        // Allocate flow using available paths in parallel
        #pragma omp parallel
        {
            #pragma omp critical
            {
                for (int i = 0; i < allPaths.size(); ++i) {
                    if (remainingDemand <= 0) continue;

                    // Calculate bottleneck capacity
                    double pathCapacity = INF;
                    for (int j = 0; j < allPaths[i].size() - 1; ++j) {
                        Edge& edge = graph.getEdge(allPaths[i][j], allPaths[i][j + 1]);
                        pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
                    }

                    if (pathCapacity > 0) {
                        moreFlowNeeded = true;

                        // Allocate flow
                        double flowToSend = min(remainingDemand, pathCapacity);

                        // Allocate and print the path in the critical section
                        sendFlow(graph, allPaths[i], flowToSend);

                        // Update demand and track delivered flow
                        unitsDelivered[lowestIndex] += flowToSend;
                        remainingDemand -= flowToSend;

                        // Debugging
                        cout << "Thread " << omp_get_thread_num() << " processed path " << i + 1 << ": ";
                        for (const auto& node : allPaths[i]) {
                            cout << node << " ";
                        }
                        cout << endl;
                    }
                }
            }
        }
    }

    cout << "\nFinal Results: Units Successfully Reaching Destinations (Before Redistribution)\n";
    for (int i = 0; i < commodities.size(); ++i) {
        cout << "Commodity " << i + 1 << " (From " << commodities[i].first
            << " to " << commodities[i].second << "): "
            << unitsDelivered[i] << "/" << demands[i] << " units\n";
    }

    // Redistribute flow after all paths are processed
    OMP_redistributeFlowForEqualization(graph, commodities, demands, unitsDelivered, successRates);

    cout << "\nFinal Results: Units Successfully Reaching Destinations (After Redistribution)\n";
    for (int i = 0; i < commodities.size(); ++i) {
        cout << "Commodity " << i + 1 << " (From " << commodities[i].first
            << " to " << commodities[i].second << "): "
            << unitsDelivered[i] << "/" << demands[i] << " units\n";
    }
}
