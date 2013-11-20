from __future__ import division
import sys
import os
import numpy as np
import ipdb
import csv
import math
from random import gauss

input_file = sys.argv[1]

class PythonHMM(object):
    """
    Main HMM class
    """
    def __init__(self,num_states):
        """
        Initialize the Model
        --> Set up data matrix
        --> Initialize Transition probabilities
        --> Initialize Emission probabilities
        --> Initialize starting and stopping weighs 
        """        
        self.data = {}
        self.data_matrix = None
        self.K = num_states # number of states
        self.state_transition_mat = {}  # The transition probability matrix ! The Matrix is a dict here !!
        self.state_symbol_prob = {}  # The emission probability sequences !
        self.state_initial_prob= {} #The initial state probability distributions !
        
        self.forward_trellis = {} # Chart for forwrad trellis
        self.backward_trellis = {} # Chart for backward Trellis
        self.posterior_state_trellis = {} # Posterior probability of each state at time
        self.posterior_transition_trellis = {} # Posterior probability for each of the state transitions !

        self.forward_scaling_vector = {} # Forward scaling factors indexed by time intervals !
        self.backward_scaling_vector = {} # Backward Scaling factors !!

        self.model = {} # The trained HMM model !!
        self.N = 0 # total number of instances !
        self.T = 0 # Total number of time slots !

        # Initialize Corpus
        self._corpusReader(input_file)
        self._initializeMatrices()
                
    def _corpusReader(self,input_file):
        """
        Read from data corpus and initialize training data
        """
        reader = csv.reader(open(input_file,"rb"))                
        temp_mat = []
        for i,each in enumerate(reader):
            tmp_list = []
            each_instance_values = each
            each_instance_values = map(lambda v:float(v), map(lambda s: s.strip(),each_instance_values))
            for j,each_index in enumerate(each_instance_values):
                tmp_list.append(each_index)                                
                self.data[(i+1,j+1)] = each_index           
            temp_mat.append(tmp_list)
            self.N = i

        self.N=self.N+1
        self.data_matrix = np.matrix(temp_mat)                
        self.T = self.data_matrix.shape[1]

    def _normpdf(self,symbol,state):
        mean = self.state_symbol_prob[state]['mean']
        sd = self.state_symbol_prob[state]['std']
        var = float(sd)**2
        pi = 3.1415926
        denom = (2*pi*var)**.5
        num = math.exp(-(float(symbol)-float(mean))**2/(2*var))
        return num/denom

    def _checkPosteriorTransision(self,t):
        """
        Sanity Check ! for debug purposes !! Must remove later !
        """
        s=0

        for i in range(1,self.K+1):
            for j in range(1,self.K+1):
                s = s+ self.posterior_transition_trellis[1][(t,t+1,j,i)]

        return s
            
    def _initializeMatrices(self):
        """
        Set up All matrices with default probabilities !!
        """
        K = self.K                
        # Initialize Initia
        rand_initial_prob = np.random.dirichlet(np.ones(K),size=1)
        rand_initial_prob = list(rand_initial_prob[0,:])
        for i in range(K):            
            self.state_initial_prob[i+1] = rand_initial_prob[i]

        # Initialize the transition MAtrix !
        for i in range(K):
            rand_initial_prob = np.random.dirichlet(np.ones(K),size=1)
            rand_initial_prob = list(rand_initial_prob[0,:])                        

            for j in range(K):
                self.state_transition_mat[(j+1,i+1)] = rand_initial_prob[j]


        # Initialize the symbol distribution Parameters ui and si (Assuming a numeric outputs ! Modelled using a gaussian ! withmean ui and std si)
        init_mean = np.mean(self.data_matrix)
        init_std = np.std(self.data_matrix)        
        
        for i in range(K):
            random_mean = gauss(init_mean,30)
            random_std = gauss(init_std,30)
            self.state_symbol_prob[i+1] = {'mean':random_mean, 'std' : random_std}          

    def _getCurrentPosteriorLikelihood(self):
        """
        Get the current model Posterior Likelihood 
        """        
        likelihood = 0
        T = self.T
        K= self.K        
        final_likelihood = 0
        
        for n in range(1,self.N+1):
            # Compute total Likelihood for all Instances P(x1...xn / theta) 
            tot_log_lik = 0
            tot_scale_factor = 0
            
            for i in range(1,self.K+1):        
                likelihood = self.posterior_state_trellis[n][(T,i)]
                tot_log_lik = tot_log_lik + likelihood

            try:
                total_log_lik = math.log(likelihood) 
            except ValueError:
                ipdb.set_trace()
                
            for t in range(1,self.T):
                scale_factor = self.forward_scaling_vector[n][t] 
                tot_scale_factor = tot_scale_factor + math.log(scale_factor)

            final_likelihood = final_likelihood + (tot_log_lik - tot_scale_factor)

        return final_likelihood
        
    def _updateTransitionMatrix(self):
        """
        Update the state transition Matrix after observing Multiple isntances
        """
        N = self.N
        K = self.K
        T= self.T

        for i in range(1,self.K+1):
            den = 0
            for t in range(1,self.T):
                for n in range(1,N+1):
                    den = den + self.posterior_state_trellis[n][(t,i)]
                    
            for j in range(1,self.K+1):  
                # For some state i,j
                s = 0
                for n in range(1,N+1):                    
                    for t in range(1,self.T):                                            
                        cur_prob = self.posterior_transition_trellis[n][(t,t+1,j,i)]
                        s = s+cur_prob

                # Compute total 
                self.state_transition_mat[(j,i)] = (s/den)

    def _updateInitialProbabilities(self):
        """
        Update the initial probabilities after observing multiple instances
        """        
        N = self.N
        K = self.K

        for i in range(1,self.K+1):
            s = 0
            updated_prob = 0
            for n in range(1,self.N+1):
                s = s+1
                updated_prob = updated_prob + self.posterior_state_trellis[n][(1,i)]
            self.state_initial_prob[i] = (updated_prob/s)

    def _updateSymbolDistributionVariance(self):
        # Update the Variance of the Gaussian         
        for i in range(1,self.K+1): # for each state
            num = 0
            den = 0
            cur_mean =  self.state_symbol_prob[i]['mean']
            for n in range(1,self.N+1): # for all observarions 
                for t in range(1,self.T+1): # for all time intervals !                   
                    pos_prob = self.posterior_state_trellis[n][(t,i)]                    
                    data_value =  self.data[(n,t)]
                    new_var = math.pow((data_value-cur_mean),2) 
                    num = num + (pos_prob*new_var)
                    den = den + pos_prob                    
        
            self.state_symbol_prob[i]['std'] = math.pow(num/den,0.5)

    def _updateSymbolDistributionMean(self):
        """
        Update the emission probability parameters for each of the states !!!
        """ 
        # Update the mean of the Gaussian 
        for i in range(1,self.K+1):
            num = 0
            den = 0
            for n in range(1,self.N+1):
                for t in range(1,self.T+1):
                    pos_prob = self.posterior_state_trellis[n][(t,i)]
                    data_value =  self.data[(n,t)]
                    num = num + (pos_prob*data_value)
                    den = den + pos_prob                    
        
            self.state_symbol_prob[i]['mean'] = num/den

    def computePosteriorStateDist(self):
        """
        Compute Posterior Distribution for all states for a given observation  
        """
        
        T=self.T
        K=self.K
        
        for Ni in range(1,self.N+1):            
            self.posterior_state_trellis[Ni] ={}
            
            for t in range(1,T+1):  #for each time slot !                              
                for i in range(1,K+1): # for each state ! 
                    self.posterior_state_trellis[Ni][(t,i)] = 0
                    num = self.forward_trellis[Ni][(t,i)]*self.backward_trellis[Ni][(t,i)]
                    den = 0
                    
                    for j in range(1,K+1):
                        prod = self.forward_trellis[Ni][(t,j)]*self.backward_trellis[Ni][(t,j)]
                        den = den + prod

                    prob = num/den
                    self.posterior_state_trellis[Ni][(t,i)] = prob


    def computePosteriorTransition(self):
        """
        Compute Posterior Distribution for all transitions (i,j) for a given observation !!
        """
        T=self.T
        K=self.K

        for Ni in range(1,self.N+1):            
            self.posterior_transition_trellis[Ni]={}

            # Compute Posterior transitions !
            for t in range(1,self.T):                                
                all_total =  0
                pair_prob = {}
                
                for i in range(1,K+1): # for each state ! 
                    alpha_ti = self.forward_trellis[Ni][(t,i)]                                        
                    for j in range(1,K+1): # for each state ! 
                        # Compute normalizing constant for all Possible transitions !!
                        beta_tplusone_j = self.backward_trellis[Ni][(t+1,j)]
                        p_j_i = self.state_transition_mat[(j,i)] # j/i
                        symbol_prob =  self._normpdf(self.data[(Ni,t+1)],j)

                        cur_prod = alpha_ti * beta_tplusone_j * p_j_i * symbol_prob
                        pair_prob[(j,i)]= cur_prod 
                        all_total = all_total + cur_prod
                        
                for each_pair, score in pair_prob.iteritems(): # for each state !                     
                    self.posterior_transition_trellis[Ni][(t,t+1,each_pair[0],each_pair[1])] = score/all_total             
        
    def forwardAlgorithm(self):
        """
        Compute forward Probability
        """        
        # compute forward chart for all data instances !
        for Ni in range(1,self.N+1):
            
            self.forward_trellis[Ni] ={}
            self.forward_scaling_vector[Ni] = []
            T = self.T
            K = self.K
                        
            for t in range(1,T+1):  #for each time slot !              
                tmp_vector = []                
                for i in range(1,K+1): # for each state !                                                                       
                    if t==1:
                        score = self.state_initial_prob[i] *  self._normpdf(self.data[(Ni,t)],i)
                        self.forward_trellis[Ni][(t,i)] = score                        
                        tmp_vector.append(score)
                    else:
                        self.forward_trellis[Ni][(t,i)] = 0
                        total_prob = 0
                        # for all states in time t-1
                        p_xt_i = self._normpdf(self.data[(Ni,t)],i)
                        for j in range(1,K+1):
                            prod= self.forward_trellis[Ni][(t-1,j)] * self.state_transition_mat[(i,j)]         #i/j                 
                            total_prob = total_prob + prod
                             
                        total_prob = p_xt_i * total_prob
                        self.forward_trellis[Ni][(t,i)] = total_prob
                        tmp_vector.append(total_prob)
                        
                # Perform Forward Scaling !!
                Smax = max(tmp_vector)
                Smax = 1/Smax
                self.forward_scaling_vector[Ni].append(Smax)

                # Scale the probabilities !!!
                for i in range(1,K+1):
                    self.forward_trellis[Ni][(t,i)] = Smax*self.forward_trellis[Ni][(t,i)]                    

    def backwardAlgorithm(self):
        """
        Run backward Algorithm
        """
        # Initialize all last
        for Ni in range(1,self.N+1):
            
            self.backward_trellis[Ni] ={}
            self.backward_scaling_vector[Ni] = []
            T = self.T
            K = self.K                        

            # for each Time slot !
            for t in reversed(range(1,T+1)):  #for each time slot !              

                tmp_vector = []
                for i in range(1,K+1): # for each state !       
                    if t==T:
                        self.backward_trellis[Ni][(t,i)] =1 # Stoppping probability is 1 for all last time slots !
                        tmp_vector.append(1)

                    else:
                        self.backward_trellis[Ni][(t,i)] = 0
                        total_prob = 0
                        
                        # for all states in time t-1                        
                        for j in range(1,K+1):                            
                            
                            alpha_j_next  = self.backward_trellis[Ni][(t+1,j)]
                            symb_prob = self._normpdf(self.data[(Ni,t+1)],j)
                            trans_prob = self.state_transition_mat[(j,i)]
                            cur_prob = alpha_j_next * symb_prob * trans_prob    #j/i                                             
                                        
                            total_prob = total_prob + cur_prob
                        self.backward_trellis[Ni][(t,i)] = total_prob

                        tmp_vector.append(total_prob)
                                                    
                # Perform Scaling !!
                Smax = max(tmp_vector)
                Smax = 1/Smax
                self.backward_scaling_vector[Ni].append(Smax)

                # Scale the probabilities !!!
                for i in range(1,K+1):
                    self.backward_trellis[Ni][(t,i)] = Smax*self.backward_trellis[Ni][(t,i)]           
        

    def trainHMM(self):
        """
        Main HMM train 
        """
        num_iterations = 100
        L_iterations = [] #Keep track of the likelihood across all iterations !! Must converge !!
        
        for i in range(1,num_iterations):

            print i
            # Run new Iteration !
            self.forwardAlgorithm()
            self.backwardAlgorithm()

            # Compute Posterior probabilities !!
            self.computePosteriorStateDist()
            self.computePosteriorTransition()

            # Get current Likelihood !!
            print self._getCurrentPosteriorLikelihood()           
                        
            # Update model !
            self._updateTransitionMatrix()
            self._updateInitialProbabilities()
            self._updateSymbolDistributionMean()
            self._updateSymbolDistributionVariance()
           

if __name__=="__main__":
    model = PythonHMM(5)
    model.trainHMM()
    print "Done"
    
        
        
