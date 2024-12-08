#ifndef FLOW_MANAGER_HPP
#define FLOW_MANAGER_HPP

#include <string>
#include <vector>
#include "NetworkGraph.hpp"

struct Commodity {
    std::string source;
    std::string destination;
    int demand;  // amount of vehicles or units going from source to destination

    Commodity(std::string src, std::string dest, int dem)
        : source(src), destination(dest), demand(dem) {
    }
};

class FlowManager {
public:
    void simulateFlowOverTime(std::vector<Edge>& edges, const std::vector<Commodity>& commodities, int timeSteps, const std::string& outputFile);
    static void addFlow(std::vector<Edge>& edges, const std::string& src, const std::string& dest, int flowToAdd);
    static void removeFlow(std::vector<Edge>& edges, const std::string& src, const std::string& dest, int flowToRemove);
};

#endif
