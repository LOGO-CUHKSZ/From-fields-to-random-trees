#include "graph.hpp"

Graph::~Graph() {
    // Clean up memory
    for(auto& [name, rv] : _rvs) {
        delete rv;
    }
    for(auto& factor : _factors) {
        delete factor;
    }
}

RV* Graph::rv(const std::string& name, int n_opts, 
              const std::vector<std::string>& labels,
              std::map<std::string, bool> meta,
              bool debug) {
    RV* rv = new RV(name, n_opts, labels, meta, debug);
    rv->init_lbp();
    add_rv(rv);
    return rv;
}

void Graph::add_rv(RV* rv) {
    rv->meta["pruned"] = false;
    _rvs[rv->name] = rv;
    _just_rvs.push_back(rv);
}

int Graph::remove_loner_rvs() {
    int removed = 0;
    std::vector<std::string> to_remove;
    
    for(const auto& [name, rv] : _rvs) {
        if(rv->n_edges() == 0) {
            rv->meta["pruned"] = true;
            to_remove.push_back(name);
            removed++;
        }
    }
    
    for(const auto& name : to_remove) {
        delete _rvs[name];
        _rvs.erase(name);
    }
    
    return removed;
}

Factor* Graph::factor(std::vector<RV*> rvs, 
                     const std::string& name,
                     const Eigen::MatrixXd* potential,
                        const double& weight,
                     std::map<std::string, bool> meta,
                     bool debug) {
    // Convert string RV names to RV pointers if needed
    for(auto& rv : rvs) {
        if(rv == nullptr) {
            throw std::runtime_error("Invalid RV pointer");
        }
    }
    // std::cout << "In factor " << weight <<std::endl;
    Factor* f = new Factor(rvs, name, potential, weight, meta, debug);
    f->init_lbp();
    add_factor(f);
    if(rvs.size() == 1){
        // std::cout << rvs[0]->name << " " << potential->size() << std::endl;
        add_node_factor(rvs[0]->name, f);
    }
    return f;
}

double Graph::joint(const std::map<std::string, int>& x) const {
    if(debug) {
        // Check assignments
        assert(x.size() == _rvs.size());
        
        for(const auto& [name, label] : x) {
            auto it = _rvs.find(name);
            assert(it != _rvs.end());
            assert(it->second->has_label(label));
        }
        
        for(const auto& [name, rv] : _rvs) {
            assert(x.find(name) != x.end());
            assert(rv->has_label(x.at(name)));
        }
    }
    
    double prod = 1.0;
    for(const auto& f : _factors) {
        prod *= f->eval(x);
    }
    return prod;
}

std::pair<std::map<std::string, int>, double> Graph::bf_best_joint() {
    std::map<std::string, int> assigned;
    std::vector<RV*> todo;
    for(const auto& [name, rv] : _rvs) {
        todo.push_back(rv);
    }
    return _bf_bj_recurse(assigned, todo);
}

std::pair<std::map<std::string, int>, double> Graph::_bf_bj_recurse(
    std::map<std::string, int>& assigned,
    std::vector<RV*> todo) {
    
    if(todo.empty()) {
        return {assigned, joint(assigned)};
    }
    
    // Try all options for first RV
    RV* rv = todo.front();
    todo.erase(todo.begin());
    
    std::map<std::string, int> best_a;
    double best_r = 0.0;
    bool first = true;
    
    for(int val = 0; val < rv->n_opts; ++val) {
        auto new_a = assigned;
        new_a[rv->name] = val;
        
        auto [full_a, r] = _bf_bj_recurse(new_a, todo);
        
        if(first || r > best_r) {
            best_r = r;
            best_a = full_a;
            first = false;
        }
    }
    
    return {best_a, best_r};
}

std::vector<RV*> Graph::_sorted_nodes() const {
    std::vector<RV*> nodes;

    // std::cout << _rvs.size() << std::endl;
    for(const auto& [name, rv] : _rvs) {
        nodes.push_back(rv);
    }

    std::sort(nodes.begin(), nodes.end(),
              [](const RV* a, const RV* b) {
                  return a->n_edges() < b->n_edges();
              });
    
    return nodes;
}

std::pair<int, bool> Graph::lbp(bool init, bool normalize,
                               int max_iters, bool progress) {
   
    auto nodes = _sorted_nodes();

    // auto nodes = _just_rvs;
    if(init) {
        init_messages(nodes);
    }

    int cur_iter = 0;
    bool converged = false;
    
    // while(cur_iter < max_iters && !converged && !E_STOP) {
    while(cur_iter < max_iters && !converged) {
        cur_iter++;

        converged = true;
        for(auto* n : nodes) {
            bool n_converged = n->recompute_outgoing(normalize);
            converged = converged && n_converged;
        }
        for(auto* n : _factors) {
            bool n_converged = n->recompute_outgoing(normalize);
            converged = converged && n_converged;
        }
        // std::cout << "Iteration " << cur_iter << " converged: " << converged << std::endl;
    }
    
    return {cur_iter, converged};
}

void Graph::init_messages(const std::vector<RV*>& nodes) {
    auto n = nodes.empty() ? _sorted_nodes() : nodes;
    for(auto* node : n) {
        node->init_lbp();
    }
}

void Graph::print_sorted_nodes() const {
    auto nodes = _sorted_nodes();
    for(const auto* node : nodes) {
        std::cout << node->name << " ";
    }
    std::cout << std::endl;
}

void Graph::print_messages(const std::vector<RV*>& nodes) const {
    auto n = nodes.empty() ? _sorted_nodes() : nodes;
    std::cout << "Current outgoing messages:" << std::endl;
    for(const auto* node : n) {
        node->print_messages();
    }
}

std::vector<std::pair<RV*, Eigen::VectorXd>> Graph::rv_marginals(
    const std::vector<RV*>& rvs,
    bool normalize) {
    std::vector<RV*> rv_list = rvs.empty() ? _sorted_nodes() : rvs;
    std::vector<std::pair<RV*, Eigen::VectorXd>> tuples;
    
    for(auto* rv : rv_list) {
        auto [marg, _] = rv->get_belief();
        
        if(normalize) {
            double sum = marg.sum();
            if(sum != 0) {
                marg /= sum;
            }

        }
        
        tuples.emplace_back(rv, marg);
    }
    
    return tuples;
}

void Graph::print_rv_marginals(const std::vector<RV*>& rvs,
                             bool normalize) const {
    std::string disp = "Marginals for RVs";
    if(normalize) {
        disp += " (normalized)";
    }
    disp += ":";
    std::cout << disp << std::endl;

    auto tuples = const_cast<Graph*>(this)->rv_marginals(rvs, normalize);
    
    for(const auto& [rv, marg] : tuples) {
        std::cout << rv->name << std::endl;
        
        std::vector<std::string> vals;
        if(rv->labels.empty()) {
            for(int i = 0; i < rv->n_opts; ++i) {
                vals.push_back(std::to_string(i));
            }
        } else {
            vals = rv->labels;
        }
        
        for(size_t i = 0; i < vals.size(); ++i) {
            std::cout << "\t" << vals[i] << "\t" << marg(i) << std::endl;
        }
    }
}

std::map<std::string, Eigen::VectorXd> Graph::get_rv_marginals() {
    

    auto tuples = const_cast<Graph*>(this)->rv_marginals(_just_rvs, true);
    std::map<std::string, Eigen::VectorXd> rtn;
    for(const auto& [rv, marg] : tuples) {
        // std::cout << rv->name << std::endl;
        rtn[rv->name] = marg;
    }
    return rtn;
}

void Graph::update_factor_marginals(std::map<std::string, Eigen::VectorXd>& node_factor_marginals){
    for (auto [node, p] : node_factor_marginals){
        // Eigen::MatrixXd mat = p.transpose();
        _node_factors[node]->set_potential(p);
    }
}