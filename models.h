#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include "SPT.h"
#include "utils.h"

#include <thread>
#include <future>
#include <pthread.h>
#include <omp.h>

#include <algorithm>
#include <utility>

#include "read_model.h"
#include <time.h>

#include<dai/alldai.h>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/functions/squared_difference.hxx>
#include <opengm/functions/absolute_difference.hxx>
#include <opengm/inference/external/libdai/mean_field.hxx>


using namespace std; 
using namespace opengm;


#ifndef C__SPT_MODELS_H
#define C__SPT_MODELS_H

class models{
public:
    size_t nx;
    size_t ny;
    int type;
    size_t numberOfLabels;
    double equal_val;
    double diff_val;
    double lambda = 0.1; 


    typedef DiscreteSpace<> Space;
    Space space;


    vector<int> nodes;
    vector<pair<int, int>> edges;
    vector<vector<int>> cliques;
    vector<vector<double>> potts;
    vector<int> state_nums;

    map<pair<int, int>, vector<vector<double>>> potentials;
    map<pair<int, int>, vector<vector<double>>> energies;
    map<vector<int>, vector<vector<double>>> clique_energies;
    map<vector<int>, vector<vector<double>>> clique_marginals;
    vector<vector<double>> node_marginals;
    vector<vector<double>> node_energies;

    typedef GraphicalModel<double, Adder, OPENGM_TYPELIST_2(ExplicitFunction<double> , PottsFunction<double> ) , Space> Model_potts;
    typedef GraphicalModel<double, Adder, OPENGM_TYPELIST_2(ExplicitFunction<double> , SquaredDifferenceFunction<float> ) , Space> Model_squared;
    typedef GraphicalModel<double, Adder, OPENGM_TYPELIST_2(ExplicitFunction<double> , AbsoluteDifferenceFunction<float> ) , Space> Model_abs;
    typedef GraphicalModel<double, Adder, ExplicitFunction<double>  , Space> Model_uai;


    typedef Model_uai::FunctionIdentifier FunctionIdentifier;
    typedef Model_uai::IndependentFactorType IndependentFactor;

    models(int nx, int ny, int numberOfLabels, int type, double equal_val, double diff_val){
        this->nx = nx;
        this->ny = ny;
        this->type = type;
        this->equal_val = equal_val;
        this->diff_val = diff_val;
        this->numberOfLabels = numberOfLabels;
        if(nx != 0){
            this->space = Space(nx*ny, numberOfLabels);
        }


    }

    inline size_t variableIndex(const size_t x, const size_t y) { 

        return y + ny * x;
    }

    void initial_nodes(int random_state=6){
        for(int i=0; i<nx*ny; i++){
            nodes.push_back(i);

            vector<double> pott;
            srand(random_state);
            for(size_t s = 0; s < numberOfLabels; ++s) {
                double v = (1.0 - lambda) * rand() / RAND_MAX;
                pott.push_back(v);
            }
            potts.push_back(pott);
            state_nums.push_back(numberOfLabels);
        }

        

    }

    void grid(){
        for(int row=0; row < nx-1; row ++){
            for(int col = 0; col < ny-1; col++){
                int node = row*nx + col;
                pair<int, int> edge1(node, node+1);
                pair<int, int> edge2(node, node+ny);
                edges.push_back(edge1);
                edges.push_back(edge2);
            
            }
        }

        for(int row=0; row < nx-1; row++){
            int node = (row + 1)* ny - 1;
            pair<int, int> edge2(node, node+ny);
            edges.push_back(edge2);
        }

        for(int col=0; col < ny-1; col++){
            int node = (nx - 1)* ny + col;
            pair<int, int> edge2(node, node+1);
            edges.push_back(edge2);
        }
    }

    void read_model(string filename){
        ifstream file;
        file.open(filename, std::ios::in);
       
    
        if (!file.is_open()){
            cout << "Model file is not found!" << endl;
            return;
        }
        string strLine;
        int row = 0;
        int node_num = 0;
        int edge_num = 0;
        int label_count = 0;


        while(getline(file,strLine))
        {   
            if(strLine.empty()){
                continue;
            }
            else{
                stringstream ss(strLine);
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
                edges.push_back(std::make_pair(e1, e2));
            }
        }
        cout << "Finish reading, get " << edges.size() << " edges." << endl;
    }

    vector<vector<double>> transpose(vector<vector<double>>& original_mat){
        vector<vector<double>> new_mat;
        for(int col=0; col < original_mat[0].size(); col++){
            vector<double> temp;
            for(int row=0; row < original_mat.size(); row++){
                temp.push_back(original_mat[row][col]);
            }
            new_mat.push_back(temp);
        }
        return new_mat;
    }

    bool read_uai(string filename){
        ifstream file;
        file.open(filename, std::ios::in);

        vector<int> nodes_with_p;
        
        if (!file.is_open()){
            cout << "Model file is not found!" << endl;
            return false;
        }
        string strLine;
        int row = 0;
        int node_num = 0;
        int edge_num = 0;
        int label_count = 0;
        
        vector<int> node_or_edge;
        int potential_idx = 0;
        int edge_idx = 0;
        vector<int> rotate;


        bool key = true;
        bool node_ = true;
        bool edge_ = true;
        int count_node = 0;
        int count_head = 0;
        while(getline(file,strLine) and key)
        {   

            if(strLine.empty()){
                row ++;
                continue;
            }

            stringstream ss(strLine);

            if(row == 0){

                cout << endl;
            }

            else if(row == 1){
                int x;
                while (ss >> x){
                    node_num = x;
                }

            }   

            else if(row == 2){
            
                int x;
                while (ss >> x){
                    state_nums.push_back(x);
                    nodes.push_back(label_count);
                    vector<double> temp_p;
                    vector<double> temp_e;
                    for(int temp_p_i=0; temp_p_i < x; temp_p_i++){
                        temp_p.push_back(1.0/x);
                        temp_e.push_back(exp(-1.0/x));
                    }

                    node_marginals.push_back(temp_p);
                    node_energies.push_back(temp_e);
                    label_count ++;
                    this->space.addVariable(x);
                }
            }

            else if(row == 3){
                int x;
                while (ss >> x){
                    edge_num = x;
                }
            }

            else if(row > 3 and row <= 3 + edge_num){
                
                int x;
                int n1;
                int n2;
                int count = 0;
                while (ss >> x){
                    if (count == 1) n1 = x;
                    else if(count == 2)n2 = x;
                    count++;
                }
                if(count > 3){
                    key = false;
                }

                if (count == 3){
                    int e1 = std::min(n1, n2);
                    int e2 = std::max(n1, n2); 
                    if(e1 == n1){rotate.push_back(0);}
                    else rotate.push_back(1);
                    edges.push_back(std::make_pair(e1, e2));
                    node_or_edge.push_back(1);
                }
                else{
                    count_node ++;
                    node_or_edge.push_back(0);
                    nodes_with_p.push_back(n1);
                }
            }
            else{
                
                int head = 0;
                double x;
                if(node_or_edge[potential_idx] == 0){
                    
                    vector<double> potential;
                    vector<double> energy;
                    double sum = 0;
                    while (ss >> x){
                        head ++;
 
                        double ev = exp(-x);
                        energy.push_back(x);
                        potential.push_back(ev);
                        sum = sum + ev;
                        
            
                    }
                    if(count_head == 0){
                        if(head == 1){
                            row++;
                            count_head ++;
                            continue;
                        }
                    }
                    else{
                        count_head = 0;
                    }
                    
                    for(int lie=0; lie < potential.size(); lie++){
                        potential[lie] = potential[lie] / sum;
                        
                    }
                    
                    node_marginals[nodes_with_p[potential_idx]] = potential;
                    node_energies[nodes_with_p[potential_idx]] = energy;
                }
                else if(node_or_edge[potential_idx] == 1){
                    
                    // cout << "Edge " << edge_idx << "Edge_num " << edge_num<< endl;
                    vector<vector<double>> potential;
                    vector<vector<double>> energy;
                    vector<double> temp1 = {};
                    vector<double> temp2 = {};
                    int col_count = 0;
                    double sum = 0;
                    while (ss >> x){
                        head ++;
  
                        // read as energy
                        double ev = exp(-x);
                        temp1.push_back(ev);
                        temp2.push_back(x);
                        sum = sum + ev;


                 
                        col_count ++;
            
                        if(rotate[edge_idx] == 0){
                            if(col_count == state_nums[edges[edge_idx].second]){
                                col_count = 0;
                                potential.push_back(temp1);
                                energy.push_back(temp2);
                                temp1 = {};
                                temp2 = {};
                            }
                        }
                        else{
                            if(col_count == state_nums[edges[edge_idx].first]){
                                col_count = 0;
                                potential.push_back(temp1);
                                energy.push_back(temp2);
                                temp1 = {};
                                temp2 = {};
                            }
                        }
                        
                    }
                    if(count_head == 0){
                        if(head == 1){
                            row++;
                            count_head ++;
                            continue;
                        }
                    }
                    else{
                        count_head = 0;
                    }
                    // if(row == 53984){cout << "here " << endl;}
                    if(rotate[edge_idx] == 1){
                       
                        potential = transpose(potential);
                        energy = transpose(energy);
                    }
                    
                    for(int hang=0; hang < potential.size(); hang++){
                        for(int lie=0; lie < potential[0].size(); lie++){
                            potential[hang][lie] = potential[hang][lie] / sum;
                           
                        }
                    }
                
                    potentials[edges[edge_idx]] = potential;
                    energies[edges[edge_idx]] = energy;
                    edge_idx ++;
                }
                potential_idx ++;
                
            }
            row ++;
        }
        
        
        
        if(key){
        
            return true;
        }
        else{
          
            return false;
        }
        
        
    }

    bool read_uai2(string filename){
        ifstream file;
        file.open(filename, std::ios::in);
       
        vector<int> nodes_with_p;
        
        if (!file.is_open()){
            cout << "Model file is not found!" << endl;
            return false;
        }
        string strLine;
        int row = 0;
        int node_num = 0;
        int edge_num = 0;
        int label_count = 0;
        
        vector<int> node_or_edge;
        int potential_idx = 0;
        int clique_idx = 0;
        vector<int> rotate;


        bool key = true;
        bool node_ = true;
        bool edge_ = true;
        int count_node = 0;
        int count_head = 0;
        while(getline(file,strLine) and key)
        {   
            // cout << row << endl;
            // skip empty line
            if(strLine.empty()){
                row ++;
                continue;
            }

            stringstream ss(strLine);

            // read model type
            if(row == 0){
                // cout << strLine << endl;
                cout << endl;
            }
            // read num of nodes
            else if(row == 1){
                int x;
                while (ss >> x){
                    node_num = x;
                }
                
                // cout << "Node num: " << node_num << endl;
            }   
            // read num of states of each node
            else if(row == 2){
            
                int x;
                while (ss >> x){
                    state_nums.push_back(x);
                    nodes.push_back(label_count);
                    vector<double> temp_p;
                    vector<double> temp_e;
                    for(int temp_p_i=0; temp_p_i < x; temp_p_i++){
                        temp_p.push_back(1.0/x);
                        temp_e.push_back(exp(-1.0/x));
                    }

                    node_marginals.push_back(temp_p);
                    node_energies.push_back(temp_e);
                    label_count ++;
                    this->space.addVariable(x);
                }
            }
            // read edge num
            else if(row == 3){
                int x;
                while (ss >> x){
                    edge_num = x;
                }
            }
            // get edges 
            else if(row > 3 and row <= 3 + edge_num){
                
                int x;
                int n1;
                int n2;
                int count = 0;
                vector<int> clique;
                while (ss >> x){
                    if (count == 1) n1 = x;
                    else if(count == 2)n2 = x;

                    if(count >=1){
                        clique.push_back(x);
                    }
                    count++;
                }
               

                if (count >= 3){

                    if(count == 3){
                        int e1 = std::min(n1, n2);
                        int e2 = std::max(n1, n2); 
                        // if(e1 == n1){rotate.push_back(0);}
                        // else rotate.push_back(1);
                        edges.push_back(std::make_pair(e1, e2));
                        // node_or_edge.push_back(1);
                    }
                    cliques.push_back(clique);
                    node_or_edge.push_back(1);
                }
                else{
                    count_node ++;
                    node_or_edge.push_back(0);
                    nodes_with_p.push_back(n1);
                }
            }
            else{
                
                int head = 0;
                double x;
                if(node_or_edge[potential_idx] == 0){
                    // cout << "Node " << potential_idx << endl;
                    vector<double> potential;
                    vector<double> energy;
                    double sum = 0;
                    while (ss >> x){
                        head ++;
                     
                        double ev = exp(-x);
                        energy.push_back(x);
                        potential.push_back(ev);
                        sum = sum + ev;

                    }
                    if(count_head == 0){
                        if(head == 1){
                            row++;
                            count_head ++;
                            continue;
                        }
                    }
                    else{
                        count_head = 0;
                    }
                    
                    for(int lie=0; lie < potential.size(); lie++){
                        potential[lie] = potential[lie] / sum;
                        
                    }
                    
                    node_marginals[nodes_with_p[potential_idx]] = potential;
                    node_energies[nodes_with_p[potential_idx]] = energy;
                }
                else if(node_or_edge[potential_idx] == 1){

                    vector<vector<double>> potential;
                    vector<vector<double>> energy;
                    vector<double> temp1 = {};
                    vector<double> temp2 = {};
                    int col_count = 0;
                    double sum = 0;
                    while (ss >> x){
                        head ++;
                        
                        double ev = exp(-x);
                        temp2.push_back(x);
                        temp1.push_back(ev);
                        sum = sum + x;

                        // col_count ++;
                        
                    //     if(rotate[edge_idx] == 0){
                    //         if(col_count == state_nums[edges[edge_idx].second]){
                    //             col_count = 0;
                    //             potential.push_back(temp1);
                    //             energy.push_back(temp2);
                    //             temp1 = {};
                    //             temp2 = {};
                    //         }
                    //     }
                    //     else{
                    //         if(col_count == state_nums[edges[edge_idx].first]){
                    //             col_count = 0;
                    //             potential.push_back(temp1);
                    //             energy.push_back(temp2);
                    //             temp1 = {};
                    //             temp2 = {};
                    //         }
                    //     }
                        
                    }
                    energy.push_back(temp2);
                    if(count_head == 0){
                        if(head == 1){
                            row++;
                            count_head ++;
                            continue;
                        }
                    }
                    else{
                        count_head = 0;
                    }
                    
                    
                    for(int hang=0; hang < potential.size(); hang++){
                        for(int lie=0; lie < potential[0].size(); lie++){
                            potential[hang][lie] = potential[hang][lie] / sum;
                           
                        }
                    }
                
                    // potentials[edges[edge_idx]] = potential;
                    clique_marginals[cliques[clique_idx]] = potential;
                    clique_energies[cliques[clique_idx]] = energy;
                    clique_idx ++;
                }
                potential_idx ++;
                
            }
            row ++;
        }
        
        
        
       return true;
        
        
    }

    void complete(){
        for(int i=0; i<nx*ny; i++){
            for(int j=i+1; j<nx*ny; j++){
                pair<int, int> edge(i, j);
                edges.push_back(edge);
            }
        }
    }


    Model_potts PottsModel(){
        Model_potts gm(space);

        for(size_t x = 0; x < nx; ++x){
            for(size_t y = 0; y < ny; ++y) {
                // function
                const size_t shape[] = {numberOfLabels};
                ExplicitFunction<double> f(shape, shape + 1);
                for(size_t s = 0; s < numberOfLabels; ++s) {

                    f(s) = potts[y + ny * x][s];
                }
                Model_potts::FunctionIdentifier fid = gm.addFunction(f);

                size_t variableIndices[] = {variableIndex(x, y)};

                gm.addFactor(fid, variableIndices, variableIndices + 1);
                
            }
        }

        PottsFunction<double> f(numberOfLabels, numberOfLabels, equal_val, diff_val);
        Model_potts::FunctionIdentifier fid = gm.addFunction(f);

      
        int count_factor = 0;
        for(auto edge : edges){
            int x1 = edge.first;
            int x2 = edge.second;
            int variableIndices[] = {x1, x2};
            sort(variableIndices, variableIndices + 2);
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }

        return gm;
    }


    Model_abs AbsModel(){
        Model_abs gm(space);
        // add unary functions
        for(size_t x = 0; x < nx; ++x)
        for(size_t y = 0; y < ny; ++y) {
            
            const size_t shape[] = {numberOfLabels};
            ExplicitFunction<double> f(shape, shape + 1);
                
            for(size_t s = 0; s < numberOfLabels; ++s) {
                f(s) = potts[y + ny * x][s];
            }
            Model_abs::FunctionIdentifier fid = gm.addFunction(f);

            size_t variableIndices[] = {variableIndex(x, y)};
            gm.addFactor(fid, variableIndices, variableIndices + 1);

        }
        
        AbsoluteDifferenceFunction<float> f(numberOfLabels, numberOfLabels, 1);
        Model_abs::FunctionIdentifier fid = gm.addFunction(f);

        int count_factor = 0;
        for(auto edge : edges){
            int x1 = edge.first;
            int x2 = edge.second;
            int variableIndices[] = {x1, x2};
            sort(variableIndices, variableIndices + 2);
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }
      
        return gm;
    }


    Model_squared SquaredModel(){
        Model_squared gm(space);
        // add unary functions
        for(size_t x = 0; x < nx; ++x)
        for(size_t y = 0; y < ny; ++y) {
            // function
            vector<double> pott;
            const size_t shape[] = {numberOfLabels};
            ExplicitFunction<double> f(shape, shape + 1);
                
            for(size_t s = 0; s < numberOfLabels; ++s) {
                f(s) = potts[y + ny * x][s];
            }
            Model_squared::FunctionIdentifier fid = gm.addFunction(f);

           
            size_t variableIndices[] = {variableIndex(x, y)};
            gm.addFactor(fid, variableIndices, variableIndices + 1);
        }
        

        SquaredDifferenceFunction<float> f(numberOfLabels, numberOfLabels, 1); 
        Model_squared::FunctionIdentifier fid = gm.addFunction(f);

        for(auto edge : edges){
            int x1 = edge.first;
            int x2 = edge.second;
            int variableIndices[] = {x1, x2};
            sort(variableIndices, variableIndices + 2);
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }
       
        return gm;
    }


    Model_uai UaiModel(){
        Model_uai gm(space);

        for(int x: nodes){

            // function
            vector<double> pott;
            const int shape[] = {state_nums[x]};
            ExplicitFunction<double> f(shape, shape + 1);
                
            for(size_t s = 0; s < state_nums[x]; ++s) {
                f(s) = node_energies[x][s];
            }
            Model_uai::FunctionIdentifier fid = gm.addFunction(f);

            int variableIndices[] = {x};
            gm.addFactor(fid, variableIndices, variableIndices + 1);
        }

        for(auto edge : edges){
            int x1 = edge.first;
            int x2 = edge.second;
            int shape[] = {state_nums[x1], state_nums[x2]};
            ExplicitFunction<double> f(shape, shape+2, 1); 

            for(int i=0; i<state_nums[x1]; i++){
                for(int j=0; j<state_nums[x2]; j++){

                    f(i, j) = energies[edge][i][j];
                }
            }

            Model_uai::FunctionIdentifier fid = gm.addFunction(f);
            int variableIndices[] = {x1, x2};
            sort(variableIndices, variableIndices + 2);
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }

        // cout << "FInish modeling!" << endl;
        return gm;
    }

    Model_uai UaiModel2() {
        Model_uai gm(space);

        for (int x : nodes) {
            vector<double> pott;
            const int shape[] = {state_nums[x]};
            ExplicitFunction<double> f(shape, shape + 1);

            for (size_t s = 0; s < state_nums[x]; ++s) {
                f(s) = node_energies[x][s];
            }

            Model_uai::FunctionIdentifier fid = gm.addFunction(f);

            
            int variableIndices[] = {x};
            gm.addFactor(fid, variableIndices, variableIndices + 1);
        }

       
        for (const auto& clique : cliques) {

            int numNodes = clique.size();
            

            vector<int> shape(numNodes);
            for (int i = 0; i < numNodes; ++i) {
                shape[i] = state_nums[clique[i]];

            }

            ExplicitFunction<double> f(shape.begin(), shape.end());


            if(numNodes == 2){
                for (int i = 0; i < state_nums[clique[0]]; ++i) {
                    for (int j = 0; j < state_nums[clique[1]]; ++j) {
                        f(i, j) = clique_energies[clique][0][i*state_nums[clique[1]] + j];
                    }
                }
            }
            else if(numNodes == 3){
                for(int i = 0; i < state_nums[clique[0]]; ++i){
                    for(int j = 0; j < state_nums[clique[1]]; ++j){
                        for(int k = 0; k < state_nums[clique[2]]; ++k){
                            f(i, j, k) = clique_energies[clique][0][i*state_nums[clique[1]]*state_nums[clique[2]] + j*state_nums[clique[2]] + k];
                        }
                    }
                }
                cout << "Yes" << endl;
            }
            else if(numNodes == 4){
                for(int i = 0; i < state_nums[clique[0]]; ++i){
                    for(int j = 0; j < state_nums[clique[1]]; ++j){
                        for(int k = 0; k < state_nums[clique[2]]; ++k){
                            for(int l = 0; l < state_nums[clique[3]]; ++l){
                                f(i, j, k, l) = clique_energies[clique][0][i*state_nums[clique[1]]*state_nums[clique[2]]*state_nums[clique[3]] + j*state_nums[clique[2]]*state_nums[clique[3]] + k*state_nums[clique[3]] + l];
                                // cout << clique_energies[clique][0][i*state_nums[clique[1]]*state_nums[clique[2]]*state_nums[clique[3]] + j*state_nums[clique[2]]*state_nums[clique[3]] + k*state_nums[clique[3]] + l] << " ";
                            }
                        }
                    }
                }
                
            }
            else if(numNodes == 5){
                for(int i = 0; i < state_nums[clique[0]]; ++i){
                    for(int j = 0; j < state_nums[clique[1]]; ++j){
                        for(int k = 0; k < state_nums[clique[2]]; ++k){
                            for(int l = 0; l < state_nums[clique[3]]; ++l){
                                for(int m = 0; m < state_nums[clique[4]]; ++m){
                                    f(i, j, k, l, m) = clique_energies[clique][0][i*state_nums[clique[1]]*state_nums[clique[2]]*state_nums[clique[3]]*state_nums[clique[4]] + j*state_nums[clique[2]]*state_nums[clique[3]]*state_nums[clique[4]] + k*state_nums[clique[3]]*state_nums[clique[4]] + l*state_nums[clique[4]] + m];
                                }
                            }
                        }
                    }
                }
            }
           
           

            Model_uai::FunctionIdentifier fid = gm.addFunction(f);

            vector<int> variableIndices(clique.begin(), clique.end());
            sort(variableIndices.begin(), variableIndices.end());

            gm.addFactor(fid, variableIndices.data(), variableIndices.data() + numNodes);
        }

    return gm;
}

    SPT* SPT_model(){


        int tree_ = 1;
        int iter_ = 1;
        int has = false;
        vector<int> obs1;
        SPT* spt = new SPT(nodes, edges, state_nums, tree_, iter_, has, obs1, potts);
        spt->initialization();
        return spt;
    }


    double uai_lost(vector<int> x, bool spt_=false){
        double lost = 0;
        for(auto edge:edges){
            int x1 = edge.first;
            int x2 = edge.second;
            int v1 = x[x1];
            int v2 = x[x2];

            if(spt_){
                v1 --;
                v2 --;
            } 
            if(!potentials[edge].empty()){
                lost = lost + energies[edge][v1][v2];

            }
        }
        for(auto node : nodes){
            int v = x[node];
            if(spt_){
                v --;
            }

            lost = lost + node_energies[node][v];

        }
        return lost;
    }

    double uai_lost_new(vector<int> x, bool spt_=false){
        double lost = 0;
        for(auto clique:cliques){
            int x1 = clique[0];
            int x2 = clique[1];
            int v1 = x[x1];
            int v2 = x[x2];

            if(spt_){
                v1 --;
                v2 --;
            } 
        
            if(clique_energies.count(clique) == 0){
                vector<int> nss = {x2, x1};
                lost = lost + clique_energies[nss][0][v2*state_nums[x1] + v1];

            }
            else{
                lost = lost + clique_energies[clique][0][v1*state_nums[x2] + v2];
            }
        }
        for(auto node : nodes){
            int v = x[node];
            if(spt_){
                v --;
            }
          
            lost = lost + node_energies[node][v];
         
        }
        return lost;
    }


    // template <typename... Functions>
    template <typename ValueType, typename OperatorType, typename FunctionList, typename SpaceType>
    vector<int> LBP_potts(GraphicalModel<ValueType, OperatorType, FunctionList, SpaceType>& gm, 
                    size_t maxNumberOfIterations, double convergenceBound, double damping, bool visual=false){
        size_t variableIndex = 0;
        vector<int> X_line;
        // // set up the optimizer (loopy belief propagation)

        cout << "LBP" << endl;
        typedef BeliefPropagationUpdateRules<Model_potts, opengm::Minimizer> UpdateRules;
        typedef MessagePassing<Model_potts, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;

        BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
        BeliefPropagation bp(gm, parameter);
        
        // optimize (approximately)
        BeliefPropagation::VerboseVisitorType visitor;
        if(visual){
            bp.infer(visitor);
        }
        else{
            bp.infer();
        }
        // bp.infer(visitor);
        // bp.infer();

        // obtain the (approximate) argmin
        vector<size_t> labeling(nx * ny);
        bp.arg(labeling);

        for(size_t y = 0; y < ny; ++y) {
            for(size_t x = 0; x < nx; ++x) {
                X_line.push_back(labeling[variableIndex]+1);
                // cout << labeling[variableIndex] << ' ';
                ++variableIndex;
            }   
            // cout << endl;
        }
        return X_line;
    }


    template <typename ValueType, typename OperatorType, typename FunctionList, typename SpaceType>
    vector<int> LBP_squared(GraphicalModel<ValueType, OperatorType, FunctionList, SpaceType>& gm, 
                    size_t maxNumberOfIterations, double convergenceBound, double damping, bool visual=false){
        size_t variableIndex = 0;
        vector<int> X_line;
        // // set up the optimizer (loopy belief propagation)

        cout << "LBP" << endl;
        typedef BeliefPropagationUpdateRules<Model_squared, opengm::Minimizer> UpdateRules;
        typedef MessagePassing<Model_squared, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;

        BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
        BeliefPropagation bp(gm, parameter);
        
        // optimize (approximately)
        BeliefPropagation::VerboseVisitorType visitor;
        if(visual){
            bp.infer(visitor);
        }
        else{
            bp.infer();
        }
        // bp.infer(visitor);
        // bp.infer();

        // obtain the (approximate) argmin
        vector<size_t> labeling(nx * ny);
        bp.arg(labeling);

        for(size_t y = 0; y < ny; ++y) {
            for(size_t x = 0; x < nx; ++x) {
                X_line.push_back(labeling[variableIndex]+1);
                // cout << labeling[variableIndex] << ' ';
                ++variableIndex;
            }   
            // cout << endl;
        }
        return X_line;
    }

    template <typename ValueType, typename OperatorType, typename FunctionList, typename SpaceType>
    vector<int> LBP_abs(GraphicalModel<ValueType, OperatorType, FunctionList, SpaceType>& gm, 
                    size_t maxNumberOfIterations, double convergenceBound, double damping, bool visual=false){
        size_t variableIndex = 0;
        vector<int> X_line;
        // // set up the optimizer (loopy belief propagation)

        cout << "LBP" << endl;
        typedef BeliefPropagationUpdateRules<Model_abs, opengm::Minimizer> UpdateRules;
        typedef MessagePassing<Model_abs, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;

        BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
        BeliefPropagation bp(gm, parameter);
        
        // optimize (approximately)
        BeliefPropagation::VerboseVisitorType visitor;
        if(visual){
            bp.infer(visitor);
        }
        else{
            bp.infer();
        }
        // bp.infer(visitor);
        // bp.infer();

        // obtain the (approximate) argmin
        vector<size_t> labeling(nx * ny);
        bp.arg(labeling);

        for(size_t y = 0; y < ny; ++y) {
            for(size_t x = 0; x < nx; ++x) {
                X_line.push_back(labeling[variableIndex]+1);
                // cout << labeling[variableIndex] << ' ';
                ++variableIndex;
            }   
            // cout << endl;
        }
        return X_line;
    }

    template <typename ValueType, typename OperatorType, typename FunctionList, typename SpaceType>
    vector<int> LBP_uai(GraphicalModel<ValueType, OperatorType, FunctionList, SpaceType>& gm, 
                    size_t maxNumberOfIterations, double convergenceBound, double damping, bool visual=false){
        size_t variableIndex = 0;
        vector<int> X_line;
        // // set up the optimizer (loopy belief propagation)

        cout << "LBP" << endl;
        typedef BeliefPropagationUpdateRules<Model_uai, opengm::Minimizer> UpdateRules;
        typedef MessagePassing<Model_uai, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;
    
        BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
        BeliefPropagation bp(gm, parameter);
        
        // optimize (approximately)
        BeliefPropagation::VerboseVisitorType visitor;

        if(visual){
            bp.infer(visitor);
        }
        else{
            bp.infer();
        }
        // bp.infer(visitor);
        // bp.infer();

        // obtain the (approximate) argmin
        vector<size_t> labeling(nodes.size());
        bp.arg(labeling);

        for (int i = 0; i < nodes.size(); i++){
            X_line.push_back(labeling[nodes[i]]+1);
        }
     
        return X_line;
    }


    template <typename ValueType, typename OperatorType, typename FunctionList, typename SpaceType>
    vector<int> TRBP_potts(GraphicalModel<ValueType, OperatorType, FunctionList, SpaceType>& gm, 
                    size_t maxNumberOfIterations, double convergenceBound, double damping, bool visual=false){
        cout << "TRBP" << endl;

        typedef TrbpUpdateRules<Model_potts, Minimizer> UpdateRules2; 
        typedef MessagePassing<Model_potts, Minimizer, UpdateRules2, MaxDistance> TRBP; 

        size_t variableIndex = 0;
        vector<int> X_line;

        TRBP::Parameter parameter(maxNumberOfIterations, convergenceBound, damping); 
        TRBP trbp(gm, parameter); 


        TRBP::VerboseVisitorType visitor; 
        if(visual){
            trbp.infer(visitor); 
        }
        else{
            trbp.infer();
        }
        

        // obtain the (approximate) argmax 
        vector<size_t> labeling(nx*ny); 
        trbp.arg(labeling);


        for(size_t y = 0; y < ny; ++y) {
            for(size_t x = 0; x < nx; ++x) {
                X_line.push_back(labeling[variableIndex]+1);
                // cout << labeling[variableIndex] << ' ';
                ++variableIndex;
            }   
            // cout << endl;
        }

        return X_line;
    }

    template <typename ValueType, typename OperatorType, typename FunctionList, typename SpaceType>
    vector<int> TRBP_squared(GraphicalModel<ValueType, OperatorType, FunctionList, SpaceType>& gm, 
                    size_t maxNumberOfIterations, double convergenceBound, double damping, bool visual=false){
        cout << "TRBP" << endl;

        typedef TrbpUpdateRules<Model_squared, Minimizer> UpdateRules2; 
        typedef MessagePassing<Model_squared, Minimizer, UpdateRules2, MaxDistance> TRBP; 

        size_t variableIndex = 0;
        vector<int> X_line;

        TRBP::Parameter parameter(maxNumberOfIterations, convergenceBound, damping); 
        TRBP trbp(gm, parameter); 
        // optimize (approximately) 

        TRBP::VerboseVisitorType visitor; 
        if(visual){
            trbp.infer(visitor); 
        }
        else{
            trbp.infer();
        }
        

        // obtain the (approximate) argmax 
        vector<size_t> labeling(nx*ny); 
        trbp.arg(labeling);


        for(size_t y = 0; y < ny; ++y) {
            for(size_t x = 0; x < nx; ++x) {
                X_line.push_back(labeling[variableIndex]+1);
                // cout << labeling[variableIndex] << ' ';
                ++variableIndex;
            }   
            // cout << endl;
        }

        return X_line;
    }


    template <typename ValueType, typename OperatorType, typename FunctionList, typename SpaceType>
    vector<int> TRBP_abs(GraphicalModel<ValueType, OperatorType, FunctionList, SpaceType>& gm, 
                    size_t maxNumberOfIterations, double convergenceBound, double damping, bool visual=false){
        cout << "TRBP" << endl;

        typedef TrbpUpdateRules<Model_abs, Minimizer> UpdateRules2; 
        typedef MessagePassing<Model_abs, Minimizer, UpdateRules2, MaxDistance> TRBP; 

        size_t variableIndex = 0;
        vector<int> X_line;

        TRBP::Parameter parameter(maxNumberOfIterations, convergenceBound, damping); 
        TRBP trbp(gm, parameter); 
        // optimize (approximately) 

        TRBP::VerboseVisitorType visitor; 
        if(visual){
            trbp.infer(visitor); 
        }
        else{
            trbp.infer();
        }


        // obtain the (approximate) argmax 
        vector<size_t> labeling(nx*ny); 
        trbp.arg(labeling);


        for(size_t y = 0; y < ny; ++y) {
            for(size_t x = 0; x < nx; ++x) {
                X_line.push_back(labeling[variableIndex]+1);
                // cout << labeling[variableIndex] << ' ';
                ++variableIndex;
            }   
            // cout << endl;
        }

        return X_line;
    }

    template <typename ValueType, typename OperatorType, typename FunctionList, typename SpaceType>
    vector<int> TRBP_uai(GraphicalModel<ValueType, OperatorType, FunctionList, SpaceType>& gm, 
                    size_t maxNumberOfIterations, double convergenceBound, double damping, bool visual=false){
        cout << "TRBP" << endl;

        typedef TrbpUpdateRules<Model_uai, Minimizer> UpdateRules2; 
        typedef MessagePassing<Model_uai, Minimizer, UpdateRules2, MaxDistance> TRBP; 

        size_t variableIndex = 0;
        vector<int> X_line;

        TRBP::Parameter parameter(maxNumberOfIterations, convergenceBound, damping); 
        TRBP trbp(gm, parameter); 
        // optimize (approximately) 

        TRBP::VerboseVisitorType visitor; 
        if(visual){
            trbp.infer(visitor); 
        }
        else{
            trbp.infer();
        }

        vector<size_t> labeling(nodes.size());
        trbp.arg(labeling);


        for (int i = 0; i < nodes.size(); i++){
            X_line.push_back(labeling[nodes[i]]+1);
        }
      
        return X_line;
    }





    vector<int> new_SPT_infer(SPT* spt, int TreeNums, int numThreads, int MAX_ITER, bool visual){
        clock_t start,end;
        clock_t start1,end1;
        vector<Graph*> Trees;
        vector<vector<pair<int, int>>> normal_trees;
        vector<vector<int>> list_of_neighbors = spt->list_of_neighbors;
        vector<vector<double>> prob_mat = spt->prob_mat;
        map<pair<int, int>, double> weights = spt->weights;
        vector<vector<double>> potts1 = spt->potts;

        
        double l1 = 99999999999.0;
        double l2 = 99999999999.0;
        double l = 99999999999.0;
        double bl = 99999999999.0;
        vector<double> Loss;
        vector<int> X;
        vector<int> X2;

        int size = nodes.size();

     
        int batch_size = TreeNums / numThreads;
        cout << "batch_size " << batch_size << endl;
        cout << "TreeNums " << TreeNums << endl;
        double merge_time = 0.0;
        double all_time = 0.0;
        // start = clock();
        for(int iter=0; iter < MAX_ITER; iter++){
            
            
            cout << "iter: " << iter << endl;
            if(iter==0){
                // start = clock();
                for(int batch=0; batch < batch_size; batch++){
                    // cout << "1 " << endl;
                    vector<thread> threads;
                    vector<Graph*> results;
                    mutex mutex;
                    for(int i=0; i < TreeNums; i++){
                        Graph* FT;
                        //parallel
                        int id = i+1+batch*batch_size;
                        threads.emplace_back(mainfunc_new, ref(nodes), ref(edges), ref(state_nums),
                                                    ref(weights), ref(list_of_neighbors), ref(prob_mat), ref(iter), ref(FT), ref(node_marginals), 
                                                    ref(id), ref(clique_energies), std::ref(results), std::ref(mutex));
                    } 
                    // cout << "2 " << endl;
                    for (auto& thread : threads) {
                        thread.join();
                    }
                    // cout << "3 " << endl;
                    for (const auto& temp : results) {
                        Trees.push_back(temp);
                    }
                    // cout << "4" << endl;
                }
                // end = clock();
            }
            
            else{
                // start = clock();
                 vector<future<void>> futures;
                for(int batch=0; batch < batch_size; batch++){
                    vector<thread> threads;
                    vector<Graph*> results;
                    mutex mutex;
                    for (int i=0; i < TreeNums; i++){

                        Graph* FT = Trees[i+batch*batch_size];
                        int id = i+batch*batch_size;
                        futures.push_back(async(launch::async, [FT]() {
                            FT->lbp(true, true, 50, true);
                        }));
                       
                    }
                    for (auto& f : futures) {
                        f.wait();
                    }

                }

                
            }            
       
            map<string, Eigen::VectorXd> vs;
            start = clock();
            for(int this_node=0; this_node < nodes.size(); this_node++){
                string key = "x"+to_string(nodes[this_node]);
                Eigen::VectorXd v = Eigen::VectorXd::Zero(state_nums[nodes[this_node]]);
                vs[key] = v;
            }
        
            for (Graph* FT : Trees){
                std::map<std::string, Eigen::VectorXd> ps = FT->get_rv_marginals();
                for(int this_node=0; this_node < nodes.size(); this_node++){
                    string key = "x"+to_string(nodes[this_node]);

                    vs[key] = vs[key] + ps[key];
                }
            }
        
            vector<vector<double>> marginals;
            for(int this_node=0; this_node < nodes.size(); this_node++){
                    string key = "x"+to_string(nodes[this_node]);
                    vs[key] = vs[key] / TreeNums;
                    std::vector<double> stdVec(vs[key].data(), vs[key].data() + vs[key].size());
                    marginals.push_back(stdVec);
            }
            
            for (Graph* FT : Trees){
                FT->update_factor_marginals(vs);
            }
    
            end = clock();
            merge_time += double(end-start)/CLOCKS_PER_SEC;
            
            X = gibs_sampling(nodes, marginals);

            X2 = greedy_sampling(nodes, marginals);


            l1 = uai_lost_new(X, true);
            l2 = uai_lost_new(X2, true);
            if(visual){
                cout << "Iteration " << iter << endl;
                

                cout << "SPT_Loss_gibbs = " << l1 << endl;
                cout << "SPT_Loss_greedy = " << l2 << endl;

            }
        
                
        }

        cout << "total merge time = " << merge_time << endl;
    
       
        return X;
    }

};

#endif //C__SPT_MODELS_H