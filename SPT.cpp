#include <iostream>
#include "SPT.h"
#include <Eigen/Dense>
#include <Eigen/QR>
#include <math.h>
#include <set>
#include <time.h>
#include <fstream>
#include <string>
#include <sstream>

#include <thread>
#include <future>


void SPT::get_prob_mat(){
    clock_t start,end;
    start = clock();
    
    std::vector<std::vector<int>>  adj_matrix;
    std::vector<std::vector<int>>  deg_mat;
    std::vector<std::vector<double>> laplacian;
    
    int size = nodes.size();

    for(int i=0; i < size; i++){
        std::vector<int> row_adj;
        std::vector<int> row_deg;

        for(int j=0; j < size; j++){
            row_adj.push_back(0);
            if(j==i) row_deg.push_back(list_of_neighbors[i].size());
            else row_deg.push_back(0);
        }
        for(int neighbor : list_of_neighbors[i]){
            row_adj[neighbor] = 1;
        }
        adj_matrix.push_back(row_adj);
        deg_mat.push_back(row_deg);
    }
    for(int i=0; i < size; i++){
        std::vector<double> row_lap;
       
        for(int j=0; j < size; j++){
            row_lap.push_back(1/size);
        }
        for(int j=0; j < size; j++){
            row_lap[j] += deg_mat[i][j] - adj_matrix[i][j];
        }
        laplacian.push_back(row_lap);
    }

    

    Eigen::MatrixXd matrix(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix(i, j) = laplacian[i][j];
        }
    }

    Eigen::MatrixXd inverseMatrix = matrix.completeOrthogonalDecomposition().pseudoInverse();
    std::vector<std::vector<double>> inv_laplacian;
    for (int i = 0; i < size; ++i) {
        std::vector<double> row;
        for (int j = 0; j < size; ++j) {
            row.push_back(inverseMatrix(i, j));
        }
        inv_laplacian.push_back(row);
    }

    
    for(int i=0; i < size; i ++){
        prob_mat.push_back(std::vector<double>(size, 0.0));
    }
    // print_double_mat(prob_mat);
    for(std::pair<int, int> edge : edges){
        std::pair<int, int> edge1(std::min(edge.second, edge.first), std::max(edge.second, edge.first));
        
        prob_mat[edge.first][edge.second] = inv_laplacian[edge.first][edge.first] + inv_laplacian[edge.second][edge.second] - inv_laplacian[edge.first][edge.second] - inv_laplacian[edge.second][edge.first];
        prob_mat[edge.second][edge.first] = prob_mat[edge.first][edge.second];
        
    }


    for(std::pair<int, int> edge : edges){
        std::pair<int, int> edge1(std::min(edge.second, edge.first), std::max(edge.second, edge.first));
       
        weights[edge1] = 1/prob_mat[edge.first][edge.second];
    }

    for(int i=0; i < size; i ++){
        double sum = 0;
        for(int j=0; j < size; j++){
            sum += prob_mat[i][j];
        }
        for(int j=0; j < size; j++){
            prob_mat[i][j] /= sum;
        }
    }
    
    
    end = clock();   
    std::cout<<"Prob mat time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;
}

void SPT::initial_tree(int id){
        clock_t start,end;
        start = clock();
        int this_root = (std::rand() % (nodes.size()-1-0+1))+ 0; 
        std::vector<std::pair<int, int>> tree = generate_spanning_trees(this_root, id, nodes.size());

        std::cout << "Tree" << std::endl;
  
        FactorTree* FT = new FactorTree();
        std::map<std::pair<int, int> , std::vector<std::vector<double>>> edge_dis;
        FT->add_nodes(nodes, state_nums);
        FT->add_edges(edges, weights, edge_dis);
        
        int pls = nodes.size();

        if(this->has_obs){
            for(int i=0; i < pls; i++){
                FT->add_node(pls+i, 1, obs[i]);
                int w = 1;
                int node2 = pls + i;
                FT->add_normal_edge(i, node2, 1);
            }
        }
        
        
        this_root = 0;
        FT->set_tree_root(this_root);
        FT->build_tree();
        FT->build_factor_tree(FT->vnodes[this_root], this_root);
        Trees[id-1] = FT;
        end = clock();   
        std::cout<<"Get Tree "<< id << " Ready in time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;

}



void SPT::initialization() {
    
    for(int i =0; i < treeNums; i++){
        Trees.push_back(new FactorTree());
    }

    get_prob_mat();


    std::cout << "SPT INITIALIZED" << std::endl;
}





std::vector<int> SPT::iteration(){
    int iter = 0;
    while(iter < iterations){
        for(FactorTree* FT : Trees){
            FT->sum_product();
        }
        std::cout << "Iteration " << iter << " Sum Product Done" << std::endl;
        update_marginals();
        std::cout << "Iteration " << iter << " Marginals Updated" << std::endl;
        loss_calculator();
        std::cout << "Iteration " << iter << " Loss Calculated" << std::endl;
        update_trees();
        std::cout << "Iteration " << iter << " Loss " << loss << std::endl;
        iter++;
    }
    return std::vector<int>();
}

void SPT::update_marginals() {


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
        
            int node = i;
            VNode* vn = FT->vnodes[node];
            
            for(int idx=0; idx < vn->marginal.size(); idx++){
                
                new_marginals[node][idx] += vn->marginal[idx];
            }
   
        }
    }

    for(int i=0; i < nodes.size(); i++){
        double sum = 0;
        for(int j=0; j < state_nums[i]; j++){
            
            marginals[i][j] = marginals[i][j] * (new_marginals[i][j]/treeNums);
            sum += marginals[i][j];
        }

        for(int idx=0; idx < state_nums[i]; idx++){
                
                marginals[i][idx] /= sum;
            }
        
        
    }
    
}


std::vector<int> SPT::gibs_sampling() {
    std::vector<int> values;
    
    for(int i =0; i < nodes.size(); i++){
        
        const std::vector<double>& marginal = marginals[i];

        float dice = rand() % (N + 1) / (float)(N + 1);
        
        float sum = 0;
        for(int j=0; j < marginal.size(); j++){
            
            if(dice <= sum + marginal[j]){
                values.push_back(j+1);
                sum += marginal[j];
                break;
            }
            else sum += marginal[j];
        }
    }

    return values;
}

void SPT::update_trees(){
    for(FactorTree* FT : Trees){
        for(int node : nodes){
            VNode* vn = FT->vnodes[node];
            vn->empty(marginals[node]);
        }
        for(FNode* fn : FT->factors){
            fn->empty();
        }
    }
}


void SPT::loss_calculator() {
    int l = 0;
    std::vector<int> new_X = gibs_sampling();
    
    X = new_X;

    for(int i=0; i < nodes.size(); i++){
        if (has_obs){
            l += pow((X[i] - obs[i]), 2);
        }
    }
    for(std::pair<int, int> edge:edges){
        int n1 = edge.first;
        int n2 = edge.second;
        
        l += pow((new_X[n1] - new_X[n2]), 2);
        
    }
   
    loss = std::min(l, loss);
}

std::vector<std::pair<int, int>> SPT::generate_spanning_trees(int root, int random_state, int num) {
    
    std::srand(random_state);
    int nb_nodes = nodes.size();
    num = (num != 0) ? num : nb_nodes;
    int n0 = 0;
    for (int i = 0; i < list_of_neighbors.size(); ++i) {
        if (!list_of_neighbors[i].empty()) {
            n0 = i;
            break;
        }
    }

    std::vector<int> state(nb_nodes, -1);
    for (int i = 0; i < list_of_neighbors.size(); ++i) {
        if (list_of_neighbors[i].empty()) {
            state[i] = 10;
        }
    }

    state[n0] = 1;
    int nb_nodes_in_tree = 1;
    std::vector<int> path;
    std::vector<std::vector<int>> branches;

    while (nb_nodes_in_tree < num) {
        
        float dice = rand() % (N + 1) / (float)(N + 1);
        double sum = 0.0;
        int n1 = -1; 
        for(int i=0; i < prob_mat[n0].size(); ++i){
            if(dice <= prob_mat[n0][i] + sum){
                n1 = i;
                break;
            }
            else sum += prob_mat[n0][i];
        }
      
        if (state[n1] == -1) {
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

