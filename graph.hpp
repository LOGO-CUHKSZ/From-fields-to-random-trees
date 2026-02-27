#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <memory>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <unordered_map>

const bool DEBUG_DEFAULT = true;
const int LBP_MAX_ITERS = 50;
// extern bool E_STOP;

// Forward declarations
class RV;
class Factor;
class Graph;


inline Eigen::VectorXd divide_safezero(const Eigen::VectorXd& a, 
                                     const Eigen::VectorXd& b) {
    Eigen::VectorXd result = Eigen::VectorXd::Ones(a.size());
    for(int i = 0; i < a.size(); ++i) {
        result(i) = (b(i) == 0) ? 1.0 : a(i) / b(i);
    }
    return result;
}

inline bool is_close(double a, double b, double rtol = 1e-5, double atol = 1e-8) {
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}



inline Eigen::MatrixXd transform_potential(int size1, int size2, std::vector<double>& potential1, std::vector<std::vector<double>>& potential2) {
    if(size2 == 0){
        Eigen::MatrixXd p(size1, 1);
        for(int i = 0; i < size1; ++i) {
            p(i, 0) = potential1[i];
        }
        return p;
    }
    else{
        Eigen::MatrixXd p(size1, size2);
        for(int i = 0; i < size1; ++i) {
            for(int j = 0; j < size2; ++j) {
                p(i, j) = potential2[0][i*size2+j];
            }
        }
        return p;
    }
    
}

class RV {
public:
    RV(const std::string& name, int n_opts, 
       const std::vector<std::string>& labels = std::vector<std::string>(),
       std::map<std::string, bool> meta = std::map<std::string, bool>(),
       bool debug = DEBUG_DEFAULT);

    std::string name;
    int n_opts;
    std::vector<std::string> labels;
    std::map<std::string, bool> meta;
    bool debug;

    std::vector<Factor*>& get_factors() { return _factors; }
    
    const std::vector<Eigen::VectorXd>& get_outgoing() const { return _outgoing; }
    
    void init_lbp();
    bool recompute_outgoing(bool normalize = false);
    int n_edges() const { return _factors.size(); }
    void attach(Factor* factor) { _factors.push_back(factor); }
    
    bool has_label(const std::string& label) const;
    bool has_label(int label) const;
    int get_int_label(const std::string& label) const;
    int get_int_label(int label) const { return label; }
    

    std::pair<Eigen::VectorXd, std::vector<Eigen::VectorXd>> get_belief() const;
    
    Eigen::VectorXd get_outgoing_for(const Factor* f) const;
    void print_messages() const;

private:
    std::vector<Factor*> _factors;
    std::vector<Eigen::VectorXd> _outgoing;
};



class Factor {
public:
    Factor(std::vector<RV*> rvs,
           const std::string& name = "",
           const Eigen::MatrixXd* potential = nullptr,
           const double& weight = 1.0,
           std::map<std::string, bool> meta = std::map<std::string, bool>(),
           bool debug = DEBUG_DEFAULT);

    std::string name;
    double weight;
    std::map<std::string, bool> meta;

    int n_edges() const { return _rvs.size(); }
    void init_lbp();
    bool recompute_outgoing(bool normalize = false);
    void attach(RV* rv);
    void set_potential(const Eigen::MatrixXd& p);

    std::pair<Eigen::MatrixXd, std::vector<Eigen::VectorXd>>  get_belief() const;
    const Eigen::MatrixXd& get_potential() const { return _potential; }
    // std::vector<RV*> get_rvs() const { return _rvs; }
    const std::vector<RV*>& get_rvs() const { return _rvs; }
    std::vector<Eigen::VectorXd> get_outgoing() const { return _outgoing; }
    Eigen::VectorXd get_outgoing_for(const RV* rv) const;
    double eval(const std::map<std::string, int>& x) const;
    void print_messages() const;

private:
    std::vector<RV*> _rvs;
    Eigen::MatrixXd _potential;
    std::vector<Eigen::VectorXd> _outgoing;
    bool debug;
};

class Graph {
public:
    Graph(bool debug = DEBUG_DEFAULT) : debug(debug) {}
    ~Graph();

    RV* rv(const std::string& name, int n_opts, 
           const std::vector<std::string>& labels = std::vector<std::string>(),
           std::map<std::string, bool> meta = std::map<std::string, bool>(),
           bool debug = DEBUG_DEFAULT);

    bool has_rv(const std::string& rv_s) const { 
        return _rvs.find(rv_s) != _rvs.end(); 
    }

    void add_rv(RV* rv);
    const std::map<std::string, RV*>& get_rvs() const { return _rvs; }
    const std::vector<Factor*>& get_factors() const { return _factors; }
    int remove_loner_rvs();

    Factor* factor(std::vector<RV*> rvs, 
                  const std::string& name = "",
                  const Eigen::MatrixXd* potential = nullptr,
                  const double& weight = 1.0,
                  std::map<std::string, bool> meta = std::map<std::string, bool>(),
                  bool debug = DEBUG_DEFAULT);

    void add_factor(Factor* factor) { _factors.push_back(factor); }
    void add_node_factor(std::string node, Factor* factor) { _node_factors[node] = factor; }
    
    double joint(const std::map<std::string, int>& x) const;
    std::pair<std::map<std::string, int>, double> bf_best_joint();
    std::pair<int, bool> lbp(bool init = true, bool normalize = false,
                            int max_iters = LBP_MAX_ITERS, bool progress = false);
    
    void init_messages(const std::vector<RV*>& nodes = std::vector<RV*>());
    void print_sorted_nodes() const;
    void print_messages(const std::vector<RV*>& nodes = std::vector<RV*>()) const;
    
    std::vector<std::pair<RV*, Eigen::VectorXd>> rv_marginals(
        const std::vector<RV*>& rvs = std::vector<RV*>(),
        bool normalize = true);
        
    void print_rv_marginals(const std::vector<RV*>& rvs = std::vector<RV*>(),
                           bool normalize = true) const;
    void debug_stats() const;
    void update_factor_marginals(std::map<std::string, Eigen::VectorXd>& node_factor_marginals);
    std::map<std::string, Eigen::VectorXd> get_rv_marginals();
private:
    std::map<std::string, RV*> _rvs;
    std::vector<RV*> _just_rvs;
    std::vector<Factor*> _factors;
    std::map<std::string, Factor*> _node_factors;
    bool debug;

    std::vector<RV*> _sorted_nodes() const;
    std::pair<std::map<std::string, int>, double> _bf_bj_recurse(
        std::map<std::string, int>& assigned,
        std::vector<RV*> todo);
};
