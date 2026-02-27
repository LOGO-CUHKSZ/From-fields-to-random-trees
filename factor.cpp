#include "graph.hpp"

Factor::Factor(std::vector<RV*> rvs,
               const std::string& name,
               const Eigen::MatrixXd* potential,
               const double& weight,
               std::map<std::string, bool> meta,
               bool debug)
    : name(name), weight(weight),meta(meta), debug(debug) {

    for(auto rv : rvs) {
        attach(rv);
    }
    
    if(potential) {
        set_potential(*potential);
    }
}



void Factor::init_lbp() {

    _outgoing.clear();

    _outgoing.resize(_rvs.size());
    
    for(size_t i = 0; i < _rvs.size(); ++i) {
        _outgoing[i] = Eigen::VectorXd::Ones(_rvs[i]->n_opts);

        _outgoing[i] /= _outgoing[i].sum();
    }
}

void Factor::attach(RV* rv) {
    rv->attach(this);
    _rvs.push_back(rv);
    _potential = Eigen::MatrixXd();  // Reset potential
}

void Factor::set_potential(const Eigen::MatrixXd& p) {
    if(debug) {
        // Check dimensions
        int total_dims = 1;
        for(auto rv : _rvs) {
            total_dims *= rv->n_opts;
        }
        assert(p.size() == total_dims && "Potential size mismatch");
    }
   
    _potential = p;
}

std::pair<Eigen::MatrixXd, std::vector<Eigen::VectorXd>> Factor::get_belief() const {

    Eigen::MatrixXd belief = _potential;
    

    std::vector<Eigen::VectorXd> incoming;
    incoming.reserve(_rvs.size());
    
    for(size_t i = 0; i < _rvs.size(); ++i) {
        incoming.push_back(_rvs[i]->get_outgoing_for(this));
    }

    

    if(_rvs.size() == 1) {
   
        belief = belief.array() * incoming[0].array();
    } else {
        for(size_t i = 0; i < _rvs.size(); ++i) {
            const Eigen::VectorXd& msg = incoming[i];


            if(i == 0) {
   
                for(int col = 0; col < belief.cols(); ++col) {
                    belief.col(col).array() *= msg.array();
                }
            } else {
        
                for(int row = 0; row < belief.rows(); ++row) {
                    belief.row(row).array() *= msg.array();
                }
            }
            
        }
    }
    
  
    double sum = belief.sum();
    if(sum > 0) {
        belief /= sum;
    }
    
    return {belief, incoming};
}

bool Factor::recompute_outgoing(bool normalize) {
    std::vector<Eigen::VectorXd> old_outgoing = _outgoing;
    auto [belief, incoming] = get_belief();
    
    bool convg = true;

    if(_rvs.size() == 1) {
        _outgoing[0] = belief;
   
        _outgoing[0] = divide_safezero(_outgoing[0], incoming[0]);
    }
    else{
        for(size_t i = 0; i < _rvs.size(); ++i) {
            if(i == 0) {
              
                _outgoing[i] = belief.rowwise().sum();
                // std::cout << "weight is " << weight << std::endl;
            } else {
           
                _outgoing[i] = belief.colwise().sum().transpose();
            }
            
          
            _outgoing[i] = divide_safezero(_outgoing[i], incoming[i]);
            
            if(normalize) {
                double sum = _outgoing[i].sum();
                if(sum > 0) {
                    _outgoing[i] /= sum;
                }
            }
        
        
            if(old_outgoing[i].size() == _outgoing[i].size()) {
                for(int j = 0; j < _outgoing[i].size(); ++j) {
                    if(!is_close(old_outgoing[i](j), _outgoing[i](j))) {
                        convg = false;
                        break;
                    }
                }
            } else {
                convg = false;
            }
        }
    }
    
    return convg;
}

Eigen::VectorXd Factor::get_outgoing_for(const RV* rv) const {
    for(size_t i = 0; i < _rvs.size(); ++i) {
        if(_rvs[i] == rv) {
            return _outgoing[i];
        }
    }
    throw std::runtime_error("RV not found");
}

double Factor::eval(const std::map<std::string, int>& x) const {

    std::vector<int> indices;
    for(auto rv : _rvs) {
        indices.push_back(x.at(rv->name));
    }
    
 
    int linear_idx = 0;
    int stride = 1;
    for(size_t i = 0; i < indices.size(); ++i) {
        linear_idx += indices[i] * stride;
        stride *= _rvs[i]->n_opts;
    }
    
    return _potential(linear_idx);
}

void Factor::print_messages() const {
    for(size_t i = 0; i < _rvs.size(); ++i) {
        std::cout << "\t" << name << " -> " << _rvs[i]->name << "\t";
        std::cout << _outgoing[i].transpose() << std::endl;
    }
}