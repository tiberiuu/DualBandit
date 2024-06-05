import statistics
import numpy as np
import pandas as pd
import math
import argparse

def softmax(softbeliefs, tau):
    # function i/o: 
    # receive a vector of beliefs (for instance [0.4, 0.3, 0.5, 0.5]) and an exploration parameter tau
    # and return a vector of probabilities that each arm should be pulled, based on the softmax formula
    # m(i) = e^(qi/tau) / sum (e^(q/tau))
    # since taus lower than 0.002 make for huge numbers and buffer overflows, make sure tau >= 0.002
    m = np.exp(softbeliefs/(max(tau,0.002)))
    return(m/sum(m))
    
def armonechoice(beliefs, myalpha, mybeta):
    """
    Function i/o:
    Receive a vector of beliefs, myalpha (epsilon for exploration/exploitation), 
    and mybeta (probability of choosing the second-best arm during exploration).

    Choose whether to explore or exploit based on myalpha (epsilon-greedy).
    If exploring: choose the second-best arm with probability mybeta, or another 
                   random arm with probability 1-mybeta, but never the best arm.
    If exploiting: pull the arm with the highest belief.

    Return the arm to pull (index).
    """

    if np.random.random() < myalpha:
        # Explore: Never choose the best arm

        # Find indices of best and second-best arms
        best_arm = np.argmax(beliefs)
        sorted_indices = np.argsort(-beliefs)  # Sort in descending order
        second_best_arm = sorted_indices[1]  # Get the second-best

        # Choose second-best with probability mybeta, otherwise random among the rest
        if np.random.random() < mybeta:
            choice = second_best_arm
        else:
            remaining_arms = np.delete(np.arange(len(beliefs)), [best_arm, second_best_arm])
            choice = np.random.choice(remaining_arms)
    else:
        # Exploit: Choose the best arm
        choice = np.argmax(beliefs)

    return choice  # Return the chosen arm index




def flatslots(myalpha, mybeta, truepayoffs):
    # function i/o:
    exitround = myplays
    prematureexit = 0
    lastarm = -1
    armschanged = 0
    
    timesplayed = np.zeros( armsboss)
    beliefsqueue = clearbeliefs.copy()
    beliefs = clearbeliefs.copy()
    sorted_indices = np.argsort(-truepayoffs)
    rankstruevalues = np.argsort(sorted_indices)
    
    # R1: reviewer request
    # instead of starting with clear beliefs, start with random beliefs
    # beliefsqueue = np.random.normal(loc=ARMMIU, scale=ARMSIGMASQ, size = armsboss)
    # beliefs = np.random.normal(loc=ARMMIU, scale=ARMSIGMASQ, size = armsboss)
    
    # R1: reviewer request
    # instead of starting with clear beliefs, start with true beliefs
    # beliefsqueue = truepayoffs.copy()
    # beliefs = truepayoffs.copy()
    
    bestarm = np.random.choice(np.where(truepayoffs == max(truepayoffs))[0],1)[0]
    
    performance = 0 
    exploration = 0
    explorationcost = 0
    knowledge_truevalue = 0

    armspullorder = np.zeros(armsboss)
    armscostorder = np.zeros(armsboss)

    lastplayed = np.zeros(armsboss)
    
    knowledgebin = 0
    
    myturbulence = turbulence/100.0
    
    # determine when we have shocks. This is firm specific (i.e., firm 1 can have shocks in
    # rounds 1, 3, and 56, while firm 2 can have shocks in rounds 23, 74 and 99)
    shock = np.random.binomial(n=1,p=myturbulence, size = myplays)
        
    for i in range(myplays):
        #
        if shock[i]:
            # In case there is a shock, redraw each arm with probability 50%, following Posen & Levinthal 2012

            newpayoffs = np.random.beta(a=2, b=2, size = armsboss)
            resetarm = np.random.binomial(n=1, p=0.5, size = armsboss)
            truepayoffs = newpayoffs * resetarm + (1-resetarm) * truepayoffs
            # just in case we have multiple best arms, pick one at random
            bestarm = np.random.choice(np.where(truepayoffs == max(truepayoffs))[0],1)[0]
            sorted_indices = np.argsort(-truepayoffs)
            rankstruevalues = np.argsort(sorted_indices)

           
        beliefs = beliefsqueue.copy()
        # reset beliefs for arms that have not been played in the last $sigmat rounds
        # this was added in the review process as a sensitivity test
        # make sigmat = myplays and it will never reset beleifs
        checkwhen = np.repeat((i-sigmat), armsboss)    
        resetbeliefs = (lastplayed < checkwhen) * 1
        beliefs = resetbeliefs * clearbeliefs + (1-resetbeliefs) * beliefs
        believedbest = np.random.choice(np.where(beliefs == max(beliefs))[0],1)[0]
        beliefsmax = beliefs.max()
        sorted_indices2 = np.argsort(-beliefs)
        ranksbeliefs= np.argsort(sorted_indices2)

        # choose what to play. $initial_budget variable shows how many plays per round. recommended value is 1, but you can set it to whatever
        # optimal tau will decrease as $initial_budget increases (because you get more feedback per round, and you don't have to explore as much)
        nextchoicevector = [armonechoice(beliefs, myalpha, mybeta) for i in range(initial_budget)]



        for zz in range(initial_budget):
            # decide what to play
            nextarm = nextchoicevector[zz]
            if (nextarm != lastarm):
                armschanged = armschanged + 1
                lastarm = nextarm
            
            #play
            thisdraw = np.random.binomial(n=1, p = truepayoffs[nextarm])

            # decay beliefs (if decay < 1)
            timesplayed[nextarm] += 1
            timesplayed = timesplayed * decay
            lastplayed[nextarm] = i
            
            #recordresult
            # 1. did i explore this pull?
            explored = (beliefsmax != beliefs[nextarm]) * 1
            exploration += explored

            # 2. add to exploration cost
            explorationcost += truepayoffs[believedbest] - truepayoffs[nextarm]
            
    
            # 3. record how much i know
            knowledge_truevalue += truepayoffs[believedbest] - truepayoffs[bestarm]

            knowledgebin += (nextarm == bestarm) * 1

            # 4. record which arm do I think i pulled in terms of value
            pullindex = ranksbeliefs[nextarm]
            armspullorder[pullindex] += 1
            
            # knowledge values not used in this paper.
            # knowledge in beliefs is 
            # belief of the actual best arm minus belief of the arm believed to be the best
            # if the best beleived arm is actually the best, knowledge = 0 (best knowledge)
            # if not, knowledge < 0 (theoretical low boundary = -1)
            #knowledge_belief += beliefs[bestarm] - beliefsmax
            
            
            ### update beliefs queue
            #
            # because we draw arms based on beliefs, and we consider "exploration" and "exploitation"
            # based on beliefs, but we only update the beliefs when the curentbudget ends. Therefore, the
            # "true" beliefs need to be kept somewhere until it's time to update them
            beliefsqueue[nextarm] = beliefsqueue[nextarm] + (1/(timesplayed[nextarm]+1)) * (thisdraw - beliefsqueue[nextarm])
            
            ### performance
            performance += thisdraw * mywin + (1-thisdraw) * mylose
            knowranks = (ranksbeliefs == rankstruevalues) * 1
            armscostorder += knowranks[sorted_indices]

            
            
        # end budget for loop
        
    #end myplays for loop
    
    

    myreturn = np.asarray((performance, exploration, armschanged, explorationcost, knowledge_truevalue+myplays, knowledgebin, statistics.pstdev(beliefsqueue), prematureexit, exitround))
    myreturn = np.hstack((myreturn, armspullorder, armscostorder))


    return(myreturn)
    #end flatslots

def playslots(args, alpha, beta, fn):
    # function i/o

    # the function uses the following global variables
    # firms = the number of firms for which to run the simulation (from command line)
    #
    gresultmatrix = [
        flatslots(alpha, beta, np.random.beta(a=2, b=2, size = armsboss))
        for i in range(args.firms)
    ]
    gresultmatrix = pd.DataFrame(gresultmatrix)
    gresultmatrix.columns = mycol1
    myresult = np.hstack(((alpha, beta, fn), gresultmatrix.mean(axis =0), gresultmatrix.std(axis =0)))
    return(myresult)
    
    #end playslots



### here are the important parameters of the code
###
# keep these up where i can see them (and eventually change them)

ARMMIU = 0.5
ARMSIGMASQ = math.sqrt(0.05)


armsboss = 10
armsigma=np.repeat(ARMSIGMASQ, armsboss)
myplays = 500

## Change here the turbulence rate 
# recommended values: 0.5, 1, 2, 4, 8, 16, 32
turbulence = 0

## Change here the memory decay rate
# recommended values 0.01, 0.03, 0.05
decayrate = 0.00
decay = 1 - decayrate


# other constants
# a very very small number
epsilon = 10 ** -7

## initial beliefs

clearbeliefs = np.repeat(0.5, armsboss)

## how much do i get for success and period
mywin = 1
mylose = -1

## sigmat is a parameter that came out of R2 request
# Following LiCalzi and Marchiori 2014 M2 model, if an arm has not been used in the past $sigmat periods reset its value to 0.5
# recommended values 20, 100
sigmat = 500

## How many pulls do I get every round
# recommended value 1
initial_budget = 1

# not used
stop_when_zero = 0

mycol0 = ['Alpha', 'Beta', 'UNUSED_PARAMETER']
mycol1_0 = ['Performance','Exploration', 'ArmsChanged', 'ExplorationCost','Knowledge_NewMeasure', 'KnowledgeBinary', 'VarBeliefs', 'Prematureexit', 'ExitRound']
mycol1_1 = ['ARMPULL_'+format(x+1) for x  in range(armsboss)]
mycol1_2 = ['ARMKnowledge_'+format(x+1) for x  in range(armsboss)]
mycol1 = mycol1_0 + mycol1_1 + mycol1_2 
mycol2 = ['SD_'+mycol1[i] for i in range(len(mycol1))]

mycol = mycol0 + mycol1 + mycol2



from multiprocessing import Pool, cpu_count
import csv


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='some magic')
    p.add_argument('firms', type=int, help='how many firms')
    args = p.parse_args()
    with open('tasklist-1.csv', newline='') as csvDataFile, Pool(cpu_count()) as p:
        jabs = p.starmap(
            playslots,
            (
                (args, float(alpha), float(beta), fn)
                for alpha, beta, fn in csv.reader(csvDataFile)
            ),
        )
    jabs = pd.DataFrame(jabs)
    jabs.columns = mycol
    jabs.to_csv('summaries/result-of-ab-softmax-tasklist-1-t'+str(turbulence)+'-decay'+str(decayrate)+'-sig'+str(sigmat)+'.csv')


