#include "graph.hpp"


RV::RV(const std::string& name, int n_opts, 
       const std::vector<std::string>& labels,
       std::map<std::string, bool> meta,
       bool debug)
    : name(name), n_opts(n_opts), labels(labels), 
      meta(meta), debug(debug) {

}


void RV::init_lbp() {

    _outgoing.clear();

    _outgoing.resize(_factors.size(), Eigen::VectorXd::Ones(n_opts));
    

    for(size_t i = 0; i < _outgoing.size(); ++i) {
        _outgoing[i] = Eigen::VectorXd::Ones(n_opts);

        _outgoing[i] /= _outgoing[i].sum();
    }
}


bool RV::recompute_outgoing(bool normalize) {
    
    std::vector<Eigen::VectorXd> old_outgoing = _outgoing;
 
    auto [total, incoming] = get_belief();

    bool convg = true;
    for(size_t i = 0; i < _factors.size(); ++i) {
     
        Eigen::VectorXd o = divide_safezero(total, incoming[i]);
        if(normalize) {
            double sum = o.sum();
            if(sum != 0) {
                o /= sum;
            }
        }
        _outgoing[i] = o;
       
        if(convg) {
            for(int j = 0; j < n_opts; ++j) {
                if(!is_close(old_outgoing[i](j), _outgoing[i](j))) {
                    convg = false;
                    break;
                }
            }
        }
    }
    return convg;
}

bool RV::has_label(const std::string& label) const {
    if(labels.empty()) {
        try {
            int val = std::stoi(label);
            return val < n_opts;
        } catch(...) {
            return false;
        }
    }
    return std::find(labels.begin(), labels.end(), label) != labels.end();
}

bool RV::has_label(int label) const {
    if(labels.empty()) {
        return label < n_opts;
    }
    return label < static_cast<int>(labels.size());
}

int RV::get_int_label(const std::string& label) const {
    if(labels.empty()) {
        return std::stoi(label);
    }
    auto it = std::find(labels.begin(), labels.end(), label);
    if(it == labels.end()) {
        throw std::runtime_error("Label not found: " + label);
    }
    return std::distance(labels.begin(), it);
}

std::pair<Eigen::VectorXd, std::vector<Eigen::VectorXd>> RV::get_belief() const {
    std::vector<Eigen::VectorXd> incoming;
    Eigen::VectorXd total = Eigen::VectorXd::Ones(n_opts);

    for(size_t i = 0; i < _factors.size(); ++i) {
        Eigen::VectorXd m = _factors[i]->get_outgoing_for(this);
        incoming.push_back(m);
        total = total.array() * m.array();
    }
    return {total, incoming};
}

Eigen::VectorXd RV::get_outgoing_for(const Factor* f) const {
    for(size_t i = 0; i < _factors.size(); ++i) {
        if(_factors[i] == f) {
            return _outgoing[i];
        }
    }
    throw std::runtime_error("Factor not found");
}

void RV::print_messages() const {
    for(size_t i = 0; i < _factors.size(); ++i) {
        std::cout << "\t" << name << " -> " << _factors[i]->name << "\t";
        std::cout << _outgoing[i].transpose() << std::endl;
    }
}