#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <string>
#include <thread>
#include <future>
#include <pthread.h>
#include "SPT.h"
#include <omp.h>
#include <time.h>

#include "graph.hpp"
using namespace std;



std::vector<std::pair<int, int>> generate_spanning_trees(std::vector<int> gnodes,
                                std::vector<std::pair<int, int>> gedges,
                                std::vector<int> gstate_nums,
                                std::vector<std::vector<double>> gprob_mat,
                                std::vector<std::vector<int>> glist_of_neighbors,
                                int root, 
                                int random_state, int num
                                ) {

    int N = 999;
    int nb_nodes = gnodes.size();
    num = (num != 0) ? num : nb_nodes;
    int n0 = 0;
    for (int i = 0; i < glist_of_neighbors.size(); ++i) {

        if (!glist_of_neighbors[i].empty()) {
            n0 = i;
            break;
        }
    }

    std::vector<int> state(nb_nodes, -1);
    for (int i = 0; i < glist_of_neighbors.size(); ++i) {
        if (glist_of_neighbors[i].empty()) {
            state[i] = 10;
        }
    }

    state[n0] = 1;
    int nb_nodes_in_tree = 1;
    std::vector<int> path;
    std::vector<std::vector<int>> branches;
    std::vector<std::pair<int, int>> b2;

    // 所有点都在树中
    while (nb_nodes_in_tree < num) {

        float dice = rand() % (N + 1) / (float)(N + 1);
        double sum = 0.0;

        int n1 = glist_of_neighbors[n0][std::rand() % glist_of_neighbors[n0].size()]; 

        if (state[n1] == -1) {
;
            path.push_back(n1);
            state[n1] = 0;
            n0 = n1;
        }

        if (state[n1] == 0) {
            int knot = -1;
            for (int i = 0; i < path.size(); ++i) {
                if (path[i] == n1) {
                    knot = i;
                    break;
                }
            }
            std::vector<int> nodes_loop(path.begin() + knot + 1, path.end());
            path.erase(path.begin() + knot + 1, path.end());
            for (int node : nodes_loop) {
                state[node] = -1;
            }
            n0 = n1;
        }

        else if (state[n1] == 1) {

            if (nb_nodes_in_tree == 1) {
                
                path.insert(path.begin(), n1);
                branches.push_back(path);
               
            }
            // 
            else {
                path.push_back(n1);
                branches.push_back(path);
            }
            
            
            for (int node : path) {
                state[node] = 1;
            }
            nb_nodes_in_tree += path.size()-1;
            std::vector<int> nodes_not_visited;
            for (int i = 0; i < state.size(); ++i) {

                if (state[i] == -1) {
                    nodes_not_visited.push_back(i);
                }
            }
            if (!nodes_not_visited.empty()) {
                n0 = nodes_not_visited[std::rand() % nodes_not_visited.size()];
                path = std::vector<int>{n0};
            }
            else{
                break;
            
            }
        }
    }
   
    std::vector<std::pair<int, int>> temp_tree_edges;
    for(std::vector<int>& branch : branches){
        for(int i=0; i < branch.size()-1; i++){
            
            temp_tree_edges.push_back(std::pair<int, int>(branch[i], branch[i+1]));
        }
    }

    
    return temp_tree_edges;
}


vector<vector<double>> updating_marginals(vector<int> nodes, vector<int> state_nums, vector<FactorTree*> Trees, int treeNums, vector<vector<double>>& marginals){
    std::vector<std::vector<double>> new_marginals;
    for(int i=0; i < nodes.size(); i++){
        std::vector<double> row;
        for(int j=0; j < state_nums[i]; j++){
            row.push_back(1);
        }
        new_marginals.push_back(row);
    }

    int tree_id = 0;
    for(FactorTree* FT : Trees){

        for(int i = 0; i<nodes.size(); i ++){
        // for(int node : nodes){
            int node = i;
            VNode* vn = FT->vnodes[node];
            
            for(int idx=0; idx < vn->marginal.size(); idx++){

                new_marginals[node][idx] *= vn->marginal[idx];
            }
            // std::cout << sum << std::endl;
            
        }
        // cout << endl;
        tree_id++;
    }


    for(int i=0; i < nodes.size(); i++){
        double sum = 0;
        bool key = true;
        for(int j=0; j < state_nums[i]; j++){
            
            if(isnan(new_marginals[i][j]) == 1){
                new_marginals[i][j] = 1.0/(state_nums[i]);
                key = false;
                break;
            }
        }

        if (key){
        
            for(int j=0; j < state_nums[i]; j++){


                marginals[i][j] = new_marginals[i][j];

                sum += marginals[i][j];
            }
            // cout << endl;
            for(int idx=0; idx < state_nums[i]; idx++){
                    marginals[i][idx] /= sum;
            }
        }
        
    }
    return marginals;
}

vector<vector<double>> updating_marginals_annealing(vector<int> nodes, vector<int> state_nums, vector<FactorTree*> Trees, int treeNums, vector<vector<double>>& marginals, int random_state, double threshold=0.3){
    std::vector<std::vector<double>> new_marginals;
    for(int i=0; i < nodes.size(); i++){
        std::vector<double> row;
        for(int j=0; j < state_nums[i]; j++){
            row.push_back(0);
        }
        new_marginals.push_back(row);
    }

    for(FactorTree* FT : Trees){

        for(int i = 0; i<nodes.size(); i ++){
        // for(int node : nodes){
            int node = i;
            VNode* vn = FT->vnodes[node];
            
            for(int idx=0; idx < vn->marginal.size(); idx++){

                new_marginals[node][idx] += vn->marginal[idx];
            }

            
        }
    }
    int N = 999;
    std::srand(random_state);
    
    int count = 0;

    for(int i=0; i < nodes.size(); i++){
        float dice = rand() % (N + 1) / (float)(N + 1);
        double sum = 0;
        bool key = true;
        for(int j=0; j < state_nums[i]; j++){
            
            if(isnan(new_marginals[i][j]) == 1){
                new_marginals[i][j] = 1.0/(state_nums[i]);
                key = false;
                break;
            }
        }
        if (key){
            
            if (dice < threshold && count < 500){
                for(int j=0; j < state_nums[i]; j++){
                    float disturb = rand() % (N + 1) / (float)(N + 1);
                    while(disturb == 0){
                        disturb = rand() % (N + 1) / (float)(N + 1);
                    }
                    marginals[i][j] = marginals[i][j] * (new_marginals[i][j]/treeNums)*disturb;
                    sum += marginals[i][j];
                }
                count ++;
            }
            else{
                for(int j=0; j < state_nums[i]; j++){
                    marginals[i][j] = marginals[i][j] * (new_marginals[i][j]/treeNums);

                    sum += marginals[i][j];
                }
            }
           
            for(int idx=0; idx < state_nums[i]; idx++){
                    marginals[i][idx] /= sum;
            }
        }
        
    }

    

    return marginals;
}


vector<int> gibs_sampling(vector<int> nodes, vector<vector<double>>& marginals) {
    vector<int> values;
    int N = 999;
    for(int i =0; i < nodes.size(); i++){
        
        const std::vector<double>& marginal = marginals[i];
        
        float dice = rand() % (N + 1) / (float)(N + 1);
        while (dice == 0){
            dice = rand() % (N + 1) / (float)(N + 1);
        }

        float sum = 0;
        bool key = true;
        int idx = 0;
        for(int j=0; j < marginal.size(); j++){
            idx = j;

            if(dice <= sum + marginal[j]){
                values.push_back(j+1);
                sum += marginal[j];
                key = false;
                break;
            }
            else sum += marginal[j];
            
        }
        if(key){
            cout << "no choice " << i << " margi size " << marginal.size() << " " <<  marginal[0]<< " " <<  marginal[1] << " dice " << dice << endl;
        }
    }


    return values;
}

vector<int> greedy_sampling(vector<int> nodes, vector<vector<double>>& marginals) {
    vector<int> values;
    int N = 999;

    for(int i =0; i < nodes.size(); i++){
        
        std::vector<double> marginal = marginals[i];
        int maxVal = -1; 
        int maxIndex = -1;

        for (int j = 0; j < marginal.size(); ++j) {
            if (marginal[j] > maxVal) {
                maxVal = marginal[j];
                maxIndex = j;
            }
        }
        values.push_back(maxIndex+1);
    }

    return values;
}


int Loss_func(vector<int>& X, vector<int>& nodes, vector<pair<int, int>>& edges, vector<int>& obs, bool has_obs) {
    int l = 0;
    
    for(int i=0; i < nodes.size(); i++){
        if (has_obs){
            l += pow((X[i] - obs[i]), 2);
        }
    }
    for(std::pair<int, int> edge:edges){
        int n1 = edge.first;
        int n2 = edge.second;
        l += pow((X[n1] - X[n2]), 2);

    }

    return l;
}

double Loss_potts(vector<int>& X, vector<int>& nodes, vector<pair<int, int>>& edges, vector<vector<double>>& potts, double& equal_val, double& diff_val){
    double l = 0;
    for(std::pair<int, int> edge:edges){
        
        int n1 = edge.first;
        int n2 = edge.second;

        if(X[n1] == X[n2]){
            l += equal_val;
        }
        else l += diff_val;
    }
    for(int node: nodes){
        l += potts[node][X[node]-1];
    }
    return l;
}


double Loss_squared(vector<int>& X, vector<int>& nodes, vector<pair<int, int>>& edges, vector<vector<double>>& potts){
    double l = 0;
   for(std::pair<int, int> edge:edges){
        int n1 = edge.first;
        int n2 = edge.second;
        l += pow((X[n1] - X[n2]), 2);
    }
    for(int node: nodes){
        l += potts[node][X[node]-1];
    }
    return l;
}

double Loss_abs(vector<int>& X, vector<int>& nodes, vector<pair<int, int>>& edges, vector<vector<double>>& potts){
    double l = 0;
   for(std::pair<int, int> edge:edges){
        int n1 = edge.first;
        int n2 = edge.second;
        l += abs((X[n1] - X[n2]));
    }
    for(int node: nodes){
        l += potts[node][X[node]-1];
    }
    return l;
}


Graph* build_trees2(vector<int>& nodes, vector<pair<int, int>>& edges, vector<int>& state_nums,  map<pair<int, int>, double>& weights,
         vector<vector<int>>& list_of_neighbors, vector<vector<double>>& prob_mat, int& iter, Graph* FT,
        vector<vector<double>>& node_marginals, int random, 
        map<vector<int>, vector<vector<double>>>& clique_energies){



    vector<pair<int, int>> tree = generate_spanning_trees(nodes, edges, state_nums, prob_mat, list_of_neighbors, 0, random, 0);

    Graph* graph = new Graph();;
    vector<RV*> rvs;
    vector<double> empty_u;
    vector<vector<double>> empty_v;

    for(int i=0; i < nodes.size(); i++){
        
        Eigen::MatrixXd p = transform_potential(state_nums[i], 0, node_marginals[i], empty_v);
        string name1 = "x" + to_string(i);
        RV* this_rv = graph->rv(name1, state_nums[i]);
        rvs.push_back(this_rv);
        string name2 = "f" + to_string(i);
        graph->factor({this_rv}, name2, &p);
    }
    for (const auto& edge : tree) {
        

        int n1 = edge.first;
        int n2 = edge.second;
        vector<int> clique = {n1, n2};

        if (clique_energies.count(clique) == 0) {
            n1 = edge.second;
            n2 = edge.first;
            clique = {n1, n2};
        }
        
        Eigen::MatrixXd p = transform_potential(state_nums[n1], state_nums[n2], empty_u, clique_energies[clique]);

        string name = "f" + to_string(n1) + "-" + to_string(n2);

        double w = weights[edge];
        if(weights[edge] == 0){
            w = weights[std::make_pair(edge.second, edge.first)];
        }
        
        
        graph->factor({rvs[n1], rvs[n2]}, name, &p, w);
    }

    return graph;

}

void message_passing_new(Graph*& FT, int& id,  mutex& mutex){
    std::lock_guard<std::mutex> lock(mutex);
    auto [iters, converged] = FT->lbp(true, true, 50, true);

}



void message_passing(FactorTree*& FT, int& id,  mutex& mutex){
    std::lock_guard<std::mutex> lock(mutex);
    FT->sum_product();

}


void update_tree_marginal(FactorTree*& FT, vector<vector<double>>& marginals, int& size, double damping=0.0){
     for(int i=0; i < size; i++){
            VNode* nod = FT->vnodes[i];
            nod->empty(marginals[i], damping);
        }
        for(FNode* fn : FT->factors){
            fn->empty();
        }

}




void mainfunc_new(vector<int>& nodes, vector<pair<int, int>>& edges, vector<int>& state_nums, 
                    map<pair<int, int>, double>& weights,
                    vector<vector<int>>& list_of_neighbors, vector<vector<double>>& prob_mat, int& iter, Graph* FT,
                    vector<vector<double>>& node_marginals, int& random, map<vector<int>, vector<vector<double>>>& clique_energies, vector<Graph*>& results, 
        mutex& mutex){
    Graph* temp = build_trees2(nodes, edges, state_nums, 
                                        weights, list_of_neighbors, prob_mat, iter, FT, node_marginals, random, clique_energies); 

    auto [iters, converged] = temp -> lbp(true, true, 20, true);

    std::lock_guard<std::mutex> lock(mutex);

    results.push_back(temp);


}
