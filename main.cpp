#include "models.h"
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <dirent.h>

#include <time.h>
#include <ctime> 

int main(){

    
    int nx = 0;
    int ny = 0;
    int numberOfLabels = 5;
    int type = 0;
    double equal_val = 0.1;
    double diff_val = 1.0;

    int treenum = 20;
    int maxiter = 40;
    int numthreads = 20;
    struct dirent *ptr;    
    DIR *dir;

    string PATH = "PATH TO YOUR FILES";

    dir=opendir(PATH.c_str()); 
    vector<string> files;
    cout << "FILE LISTS: "<< endl;

    while((ptr=readdir(dir))!=NULL)
    {
        if(ptr->d_name[0] == '.')
            continue;
        files.push_back(ptr->d_name);
    }
    int iterr = 40;
    double dampiing = 0.0;


    vector<int> treenumset = {20};
    // for(int iter=0; iter<treenumset.size(); iter++){
    for(int iter=0; iter<1; iter++){
        cout << iter << endl;
        for (int i = 0; i < 1; ++i)
        {   
            treenum = treenumset[iter];
            if(treenum < numthreads) numthreads = treenum;
            else if(treenum > numthreads && treenum > 20) numthreads = 20;
            int len = files[i].size();
            bool evid_file = files[i].substr(len-4, len) == "evid";
            bool wcsp_file = files[i].substr(0, 4) == "wcsp";
            // bool type = files[i].substr(0,3) == "rus";
            bool type = true;
            if(!evid_file && !wcsp_file && type){
               
                models* model = new models(nx, ny, numberOfLabels, type, equal_val, diff_val);
                // bool v = model->read_uai(PATH + "/" + files[i]);
                bool v = model->read_uai2(PATH + "/" + files[i]);
                if(v){

                 
                    cout << files[i] << endl;
                    
                    
                    models::Model_uai gm = model->UaiModel2();
                   

                    bool visual = true;
                    clock_t start = std::clock();

                    // model->LBP_uai(gm, iterr, 1e-3, dampiing, visual);
                    std::clock_t end = std::clock();
                    double duration = static_cast<double>(end - start) / CLOCKS_PER_SEC;
                    std::cout << "Execution time of LBP: " << duration << " seconds" << std::endl;

                    

                    start = std::clock();
                    // model->TRBP_uai(gm, iterr, 1e-3, dampiing, visual);

                    end = std::clock();
                    duration = static_cast<double>(end - start) / CLOCKS_PER_SEC;

                    
                    std::cout << "Execution time of TRBP: " << duration << " seconds" << std::endl;
                    
                    auto start2 = std::chrono::high_resolution_clock::now();
                    SPT* spt = model->SPT_model();

                    vector<int> xs = model->new_SPT_infer(spt, treenum, numthreads, maxiter,  true);
                    auto end2 = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double> duration2 = end2 - start2;
                    cout << "SPT TIME: " << duration2.count() << " seconds" << endl;
                    
                  
                    cout << endl;
                    cout << endl;
                }
            }
        }
    }

    return 0;


}