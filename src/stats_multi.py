import subprocess
import threading
import matplotlib.pyplot as plt  
import numpy as np
import os
from scipy import stats
from itertools import combinations

episodeCount = 300
trainEpisodes = 0
randomLayout = "mediumClassic"

def reinforceAgent(numOfRuns, results):
    command= [
        "python pacman.py -p ReinforceAgent -n {0} -x {1} -a alpha=0.2,gamma=0.8 -q -l {2}".format(episodeCount, trainEpisodes, randomLayout)
    ]

    for i in range(numOfRuns):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print("ReinforceAgent stderr: ", result.stderr)
        
        float_list1 = []
        for line in result.stdout.decode('utf-8').splitlines():
            if 'Scores:' in line:
                value1 = (line.split(":")[1].strip())  # Extract integer value
                value2 = (value1.split(", "))
                float_list1=[float(value) for value in value2] 

        for j in range(len(float_list1)):
            results[j] += float_list1[j]

        print(f"Finished reinforce {i}. \n\t Mean Score: ", np.mean(float_list1))

def qLearningAgent(numOfRuns, results):
    command= [
        "python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -n {0} -x {1} -q -l {2}".format(episodeCount, trainEpisodes, randomLayout)
    ]

    for i in range(numOfRuns):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print("ApproximateQAgent stderr: ", result.stderr)
        
        float_list1 = []
        for line in result.stdout.decode('utf-8').splitlines():
            if 'Scores:' in line:
                value1 = (line.split(":")[1].strip())  # Extract integer value
                value2 = (value1.split(", "))
                float_list1=[float(value) for value in value2] 

        for j in range(len(float_list1)):
            results[j] += float_list1[j]

        print(f"Finished ApproximateQAgent {i}. \n\t Mean Score: ", np.mean(float_list1))

def actorCriticAgent(numOfRuns, results):
    command= [
        "python pacman.py -p ActorCriticAgent -n {0} -x {1} -a alpha_theta=0.25,alpha_w=0.15,gamma=0.9 -q -l {2}".format(episodeCount, trainEpisodes, randomLayout)
    ]

    for i in range(numOfRuns):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print("ActorCriticAgent stderr: ", result.stderr)
        
        float_list1 = []
        for line in result.stdout.decode('utf-8').splitlines():
            if 'Scores:' in line:
                value1 = (line.split(":")[1].strip())  # Extract integer value
                value2 = (value1.split(", "))
                float_list1=[float(value) for value in value2] 

        for j in range(len(float_list1)):
            results[j] += float_list1[j]

        print(f"Finished ActorCriticAgent {i}. \n\t Mean Score: ", np.mean(float_list1))

def finalttest(data_dict, alpha=0.05):
    # Get all unique pairs of keys and their corresponding lists
    key_pairs = list(combinations(data_dict.keys(), 2))
    results = {}
    
    # Perform t-tests for each pair
    for key1, key2 in key_pairs:
        t_stat, p_value = stats.ttest_ind(data_dict[key1], data_dict[key2])
        
        pair_name = f"{key1} vs {key2}"
        results[pair_name] = {
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "message": f'There is {"a significant" if p_value < alpha else "no significant"} difference between the means of {key1} and {key2}'
        }

    return results

def main():
    numOfRuns = 200
    maxThreads = os.cpu_count()
    agentFunctions = [reinforceAgent, qLearningAgent, actorCriticAgent]
    results = []

    print("Cores: ", maxThreads, "\nRuns Per Core: ", round(numOfRuns / maxThreads))

    for function in agentFunctions:
        functionResults = [0] * episodeCount
        numOfRunsPerCore = round(numOfRuns / maxThreads)

        allocatedRuns = 0
        threads = []
        for _ in range(maxThreads):
            runs = min(numOfRunsPerCore, numOfRuns - allocatedRuns)
            thread = threading.Thread(target=function, args=(runs, functionResults))
            thread.start()
            threads.append(thread)
            allocatedRuns += runs

        for thread in threads:
            thread.join()

        for i in range(len(functionResults)):
            functionResults[i] /= numOfRuns

        results.append(functionResults)

    i=np.arange(len(results[0]))
    plt.plot(i,results[0],'r',label="REINFORCE Agent")
    plt.plot(i,results[1],'k',label="Approximate QLeaning Agent")
    plt.plot(i,results[2],'b',label="Actor Critic Agent")

    plt.title('Learning Methods Convergence')
    plt.xticks(i[::100])
    plt.xlabel('Episodes')
    plt.ylabel(f'Average Score Over {numOfRuns} runs.')
    plt.legend()
    plt.savefig('learning_methods.png')
    plt.savefig("../analysis/convergence.png")

    #################################
    # T-test Statistics
    Methods = {
        'Reinforcement': results[0],
        'QLearning': results[1],
        'Actor-Critic': results[2]
    }

    testResults = finalttest(Methods)

    for key, value in testResults.items():
        print(key)
        print("T-statistic:", value["t_stat"])
        print("P-value:", value["p_value"])
        print(value["message"])
        print("Significant:", value["significant"])
        print()


if __name__=="__main__":
    main()
