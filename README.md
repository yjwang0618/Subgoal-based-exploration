# Exploration via Sample-Efficient Subgoal Design

## What do we have
### Requirements
  * Python 2.7
  * MOE
  * Replace the corresponding files with the files in folder ```moe```, and rerun ```moe/optimal_learning/cpp/CMakeLists.txt``` and ```moe/optimal_learning/cpp/CMakeCache.txt```.

### Baseline Algorithms
  * EI and LCB
  * HyperBand
  * QL and "Transfer" QL
  * Random

## How to run the algorithms
#### BESD [run_besd_REP.py](run_besd_REP.py)
  ```bash
  (VIRT_ENV) $ python run_besd_REP.py miso_gw 0 0 gw10Two1
  ```
  This is the main file for running BESD. It requests 5 inputs:
  
  1) sys.argv[1]: The environment, takes value in [miso_gw, miso_ky, miso_it, miso_mc]
  
  2) sys.argv[2]: Which_problem, value is 0
  
  3) sys.argv[3]: Version, value is 0
  
  4) sys.argv[4]: Replication_no, takes value as 0,1,2,... 
  
  5) sys.argv[5]: Problem name: 
     
     when the environment is miso_gw, takes value in [gw10Two2, gw20Three1].
     
     when the environment is miso_ky, value is ky10One.
     
     when the environment is miso_it, value is it10.
     
     when the environment is miso_mc, value is mcf2.
  
#### EI / LCB [main_gpyopt.py](main_gpyopt.py)
  ```bash
  (VIRT_ENV) $ python main_gpyopt.py gw10Two1 EI 0 0
  ```
  1) sys.argv[1]: Problem name, takes value in [gw10Two1, gw20Three1, ky10One, it10, mcf2].
  
  2) sys.argv[2]: algorithm, takes value in [EI, LCB]
  
  3) sys.argv[3]: Version, value is 0
  
  4) sys.argv[4]: Replication_no, takes value as 0,1,2,... 

#### HyperBand [main_hb.py](main_hb.py)
  ```bash
  (VIRT_ENV) $ python main_hb.py gw10Two1 0 0
  ```
  1) sys.argv[1]: Problem name, takes value in [gw10Two1, gw20Three1, ky10One, it10, mcf2]
  
  2) sys.argv[2]: Version, value is 0
  
  3) sys.argv[3]: Replication_no, takes value as 0,1,2,... 

#### Q-learning and "Transfer" QL [main_ql.py](main_ql.py)
  ```bash
  (VIRT_ENV) $ python main_ql.py gw10Two1 0
  ```
  1) sys.argv[1]: Problem name, takes value in [gw10Two1, gw20Three1, ky10One, it10, mcf2]
  
  2) sys.argv[2]: 0 (QL) or 2 ("transfer" QL)

#### Random [main_random.py](main_random.py)
  ```bash
  (VIRT_ENV) $ python main_random.py gw10Two1 0 0
  ```
  1) sys.argv[1]: Problem name, takes value in [gw10Two1, gw20Three1, ky10One, it10, mcf2]
  
  2) sys.argv[2]: Version, value is 0

  3) sys.argv[3]: Replication_no, takes value as 0,1,2,... 