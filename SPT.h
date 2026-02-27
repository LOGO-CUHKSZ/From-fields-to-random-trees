#include<iostream>
#include "FactorTree.h"
#include <utility>
#include <vector>
#include <unordered_map>
#include <map>

#include <fstream>
#include <string>
#include <sstream>



#ifndef SPT_CPP_SPT_H
#define SPT_CPP_SPT_H



class SPT{
    
public:
    int N = 999;
    int treeNums;
    int iterations;
    std::vector<std::vector<double>> marginals;
    std::vector<int> nodes;
    std::vector<std::pair<int, int>> edges;
    std::vector<int> state_nums;
    std::vector<int> obs;
    bool has_obs;
    std::vector<FactorTree*> Trees;
    std::vector<std::vector<int>> list_of_neighbors;
    std::vector<std::vector<double>> prob_mat;
    std::map<std::pair<int, int>, double> weights;
    int loss = INT32_MAX; 
    std::vector<int> X;
    std::vector<std::vector<double>> potts;

    SPT(std::vector<int> nodes, std::vector<std::pair<int, int>>& edges, std::vector<int>& state_nums,
        int treeNums, int iterations,
        bool has_obs, std::vector<int>& obs, std::vector<std::vector<double>>& potts){
        this->nodes = nodes;
        this->edges = edges;
        this->state_nums = state_nums;
        this->treeNums = treeNums;
        this->iterations = iterations;
        this->has_obs = has_obs;
        this->obs = obs;

        for(int node: this->nodes){
            
            this->list_of_neighbors.push_back(std::vector<int>()) ;
        }

        for(std::pair<int, int> edge : this->edges){
            int n1 = edge.first;
            int n2 = edge.second;
            this->list_of_neighbors[n1].push_back(n2);
            this->list_of_neighbors[n2].push_back(n1);
        }

        for(int i=0; i < nodes.size(); i++){
            std::vector<double> marg(state_nums[i], 1.0/state_nums[i]);
            this->marginals.push_back(marg);
        }
        this->potts = potts;
    }

    void initialization();

    void get_prob_mat();


    void initial_tree(int id);



    std::vector<int> iteration();

    std::vector<std::pair<int, int>> generate_spanning_trees(
                                                    int root = -1,
                                                    int random_state = 0,
                                                    int num = 0);

    std::vector<int> gibs_sampling();

    void update_marginals();

    void loss_calculator();

    void update_trees();

    std::vector<std::vector<double>> inverse(std::vector<std::vector<double>>& mat);

    void print_int_mat(std::vector<std::vector<int>>& mat){
        for(int row = 0; row < mat.size(); row++){

            for(int col = 0; col < mat[row].size(); col++){
                std::cout << mat[row][col] << " ";
            }
             std::cout << std::endl;
        }
    }

    void print_double_mat(std::vector<std::vector<double>>& mat){
        for(int row = 0; row < mat.size(); row++){
            for(int col = 0; col < mat[row].size(); col++){
                std::cout << mat[row][col] << " ";
            }
             std::cout << std::endl;
        }
    }

    std::vector<double> vector_product(std::vector<double> vec1, std::vector<double> vec2){
        if(vec1.size() != vec2.size()){
            std::cout << "Vec Size doesn't match!" << std::endl;
            std::cout << "size 1 is "<< vec1.size() << std::endl;
            std::cout << "size 2 is "<< vec2.size() << std::endl;
            return  {0.0};
        }
        std::vector<double> rtn;
        for(int i=0; i < vec1.size(); i ++){
            rtn.push_back(vec1[i]*vec2[i]);
        }
        return rtn;
    }

    void print_vec(std::vector<int>& vec){
        std::cout << '[' ;
        for(double v: vec){
            std::cout << v << " ";
        }
        std::cout << ']' ;
        std::cout << std::endl;
    }

    std::vector<std::pair<int, int>> read_tree(int id){

        std::ifstream file;
        std::string filePath1 = ""; 
        std::string filePath2 = ""; 
    
    
        if (id == 1){
            file.open(filePath1, std::ios::in);
        }
        else{
            file.open(filePath2, std::ios::in);
        }
            
        if (!file.is_open()){
            std::cout << "Model file is not found!" << std::endl;
            
        }
        std::string strLine;
        int row = 0;
        int node_num = 0;
        int edge_num = 0;
        int label_count = 0;

        std::vector<int> tnodes;
        std::vector<std::pair<int, int>> tedges;
        std::vector<int> tstate_nums;
        std::vector<int> tobs;

        while(getline(file,strLine))
        {
            if(strLine.empty())
                continue;
            std::stringstream ss(strLine);
            if(row == 0){
                int x;
                while (ss >> x){
                    node_num = x;
                }

                for(int i = 0; i < node_num; i++){
                    tnodes.push_back(i);
                }
            }
            else if(row >= 1 and row < node_num + 1){
                
                int x;
                int count = 0;
                while (ss >> x){/
                    if(count==0) tstate_nums.push_back(x);
                    else if(count==1) tobs.push_back(x);
                    count++;
                }
                // create node
               
            }
            else if(row == node_num + 1){
                ss >> edge_num ;
                std::cout << edge_num << "Edges" << std::endl;
            }
            else{
                int x;
                int n1;
                int n2;
                int count = 0;
                while (ss >> x){
                    if (count == 0) n1 = x;
                    else n2 = x;
                    count++;
                }
                int e1 = std::min(n1, n2);
                int e2 = std::max(n1, n2);
                tedges.push_back(std::make_pair(e1, e2));

            }

            row++;
        }
        file.close();
        return tedges;
    }

};

#endif //SPT_CPP_SPT_H
