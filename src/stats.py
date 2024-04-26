from scipy import stats
import subprocess
import matplotlib.pyplot as plt  # Common import for matplotlib
import numpy as np
import random
import os
from itertools import combinations
# Generating layouts
filenames=os.listdir("layouts")
val2=[]
laylay=[]
for f in filenames:
    val2 = (f.replace(".lay",""))
    laylay.append(val2)
float_listr=[]

amountOfRuns = 1

# randomLayout=random.choice(laylay)
randomLayout = "mediumClassic"

numberOfEpisodes=500
trainEpisodes=0
command= [
    "python pacman.py -p ReinforceAgent -n {0} -x {1} -a alpha=0.2,gamma=0.8 -q -l {2}".format(numberOfEpisodes, trainEpisodes, randomLayout)
    ]

reinforceScores = [0]*numberOfEpisodes
for i in range(amountOfRuns):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print("ReinforceAgent stderr: ", result.stderr)
        
    float_list1 = []
    for line in result.stdout.decode('utf-8').splitlines():
        if 'Scores:' in line:
            value1 = (line.split(":")[1].strip())  # Extract integer value
            value2 = (value1.split(", "))
            float_list1=[float(value) for value in value2] 

    for j in range(len(float_list1)):
        reinforceScores[j] += float_list1[j]

    print(f"Finished reinforce {i}. \n\t Mean Score: ", np.mean(float_list1))

# Averages score
for j in range(len(reinforceScores)):
    reinforceScores[j] /= amountOfRuns 

###########################################        
###########################################
numberOfEpisodes=500
trainEpisodes=0

command= [
        "python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -n {0} -x {1} -q -l {2}".format(numberOfEpisodes, trainEpisodes, randomLayout)
    ]

qAgentScores = [0]*numberOfEpisodes
for i in range(amountOfRuns):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print("ApproximateQAgent stderr: ", result.stderr)
        
    float_list2 = []
    for line in result.stdout.decode('utf-8').splitlines():
        if 'Scores:' in line:
            value1 = (line.split(":")[1].strip())  # Extract integer value
            value2 = (value1.split(", "))
            float_list2=[float(value) for value in value2] 

    for j in range(len(float_list2)):
        qAgentScores[j] += float_list2[j]

    print(f"Finished ApproximateQAgent {i}. \n\t Mean Score: ", np.mean(float_list2))

# Averages score
for i in range(len(qAgentScores)):
    qAgentScores[i] /= amountOfRuns 
########################################        
########################################
numberOfEpisodes=500
trainEpisodes=0
command= [
        "python pacman.py -p ActorCriticAgent -n {0} -x {1} -a alpha_theta=0.25,alpha_w=0.15,gamma=0.9 -q -l {2}".format(numberOfEpisodes, trainEpisodes, randomLayout)
    ]

actorCriticScores = [0]*numberOfEpisodes
for i in range(amountOfRuns):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print("ActorCriticAgent stderr: ", result.stderr)
        
    float_list3 = []
    for line in result.stdout.decode('utf-8').splitlines():
        if 'Scores:' in line:
            value1 = (line.split(":")[1].strip())  # Extract integer value
            value2 = (value1.split(", "))
            float_list3=[float(value) for value in value2] 

    for j in range(len(float_list3)):
        actorCriticScores[j] += float_list3[j]

    print(f"Finished ActorCriticAgent {i}. \n\t Mean Score: ", np.mean(float_list3))

# Averages score
for i in range(len(actorCriticScores)):
    actorCriticScores[i] /= amountOfRuns 
##################################
#Plot the episodes
i=np.arange(len(reinforceScores))
plt.plot(i,reinforceScores,'r',label="REINFORCE Agent")
plt.plot(i,qAgentScores,'k',label="Approximate QLeaning Agent")
plt.plot(i,actorCriticScores,'b',label="Actor Critic Agent")

plt.title('Learning Methods Convergence')
plt.xticks(i[::100])
plt.xlabel('Episodes')
plt.ylabel(f'Average Score Over {amountOfRuns} runs.')
plt.legend()
plt.savefig('learning_methods.png')
plt.savefig("../analysis/convergence.png")

#################################
# T-test Statistics
Methods = {
    'Reinforcement': reinforceScores,
    'QLearning': qAgentScores,
    'Actor-Critic': actorCriticScores
}
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


results = finalttest(Methods)

for key, value in results.items():
    print(key)
    print("T-statistic:", value["t_stat"])
    print("P-value:", value["p_value"])
    print(value["message"])
    print("Significant:", value["significant"])
    print()