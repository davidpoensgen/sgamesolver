# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2018  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import cython

import numpy as np
cimport numpy as np






#%% auxiliary functions




## overall payoff u_tilde from instantaneous payoff u and continuation value V, 
## given discounted transition matrix phi
def u_tilde(u, V, phi):
    ## phi: indices [s,p,A,s'] named sp...S
    ## V: indices [s',p] named Sp
    ## u: [s,p,A]
    ## idea: combine phi and V to get array over [s,p,A] with axes sp...
    ## => multiply over s' (S axes), sum over s' (resulting S axis) and keep [s,p,A] (sp... axes)
    return u + np.einsum('sp...S,Sp->sp...', phi, V)




## payoffs u_tilde_{s,i,a} (including continuation value) of player i using pure action a in state s, 
## given mixed strategy profile sigma[s,p,a] of other players
@cython.boundscheck(False)
@cython.wraparound(False)
def u_tilde_sia(
        np.ndarray[np.float64_t, ndim=1] u_tilde_ravel, 
        np.ndarray[np.float64_t, ndim=3] sigma, 
        int num_s, 
        int num_p, 
        int num_a_max, 
        np.ndarray[np.int32_t, ndim=2] nums_a
        ):
    
    
    cdef: 
        np.ndarray[np.float64_t, ndim=3] out_ = np.zeros(shape=(num_s,num_p,num_a_max), dtype=np.float64)
        np.ndarray[np.int32_t, ndim=1] loop_profile = np.zeros(num_p+1, dtype=np.int32)
        ## loop_profile is used to loop over all action profiles
            ## loop_profile[1:num_p+1] gives current action profile, loop_profile[0] in {0,1} indicates whether all action profiles have been explored (1) or not (0)
            ## last element of action profile is iterated first; once that is done, increase second last element by one and set last element to zero again, and so on...
            ## continue until very first element of loop_profile is increased from zero to one, which means all action profiles have been explored
        double temp_prob 
        int state, player, other, n
        int flat_index = 0
    
    
    for state in range(num_s):
        for player in range(num_p):
            loop_profile[0:num_p+1] = 0
            while loop_profile[0] == 0:
                
                temp_prob = 1
                for other in range(num_p):
                    if other == player:
                        continue
                    temp_prob *= sigma[state,other,loop_profile[other+1]]
                
                out_[ state, player, loop_profile[player+1] ] += temp_prob * u_tilde_ravel[flat_index]
                flat_index +=1
                
                loop_profile[num_p] +=1
                for n in range(num_p):
                    if loop_profile[num_p-n] == nums_a[state,num_p-n-1]:
                        loop_profile[num_p-n-1] += 1
                        loop_profile[num_p-n] = 0
                        flat_index += (num_a_max - nums_a[state,num_p-n-1]) * num_a_max**n        
    
    return out_






## derivatives of u_tilde_{s,i,a} w.r.t. log strategies beta_{i',a'} 
    ## (no index s' in beta because corresponding derivative is zero)
@cython.boundscheck(False)
@cython.wraparound(False)   
def u_tilde_sia_partial_beta(
        np.ndarray[np.float64_t, ndim=1] u_tilde_ravel, 
        np.ndarray[np.float64_t, ndim=3] sigma, 
        int num_s, 
        int num_p, 
        int num_a_max, 
        np.ndarray[np.int32_t, ndim=2] nums_a
        ):

    
    cdef: 
        np.ndarray[np.float64_t, ndim=5] out_ = np.zeros((num_s,num_p,num_a_max,num_p,num_a_max), dtype=np.float64)
        np.ndarray[np.int32_t, ndim=1] loop_profile = np.zeros(num_p+1, dtype=np.int32)
        double temp_prob
        int state, player, player_j, other, n
        int flat_index = 0
    
    
    for state in range(num_s):
        for player in range(num_p):
            loop_profile[0:num_p+1] = 0
            while loop_profile[0] == 0:
                
                temp_prob = 1
                for other in range(num_p):
                    if other == player:
                        continue
                    temp_prob *= sigma[state,other,loop_profile[other+1]]

                for player_j in range(num_p):
                    if player_j == player:
                        continue
                    out_[ state, player, loop_profile[player+1], player_j, loop_profile[player_j+1] ] += temp_prob * u_tilde_ravel[flat_index]
                flat_index += 1 
                
                loop_profile[num_p] += 1
                for n in range(num_p):
                    if loop_profile[num_p-n] == nums_a[state,num_p-n-1]:
                        loop_profile[num_p-n-1] += 1
                        loop_profile[num_p-n] = 0
                        flat_index += (num_a_max - nums_a[state,num_p-n-1]) * num_a_max**n       
    
    return out_




## derivatives of u_tilde_{s,i,a} w.r.t. continuation value V_{s',i'}
@cython.boundscheck(False)
@cython.wraparound(False)
def u_tilde_sia_partial_V(
        np.ndarray[np.float64_t, ndim=1] phi_ravel, 
        np.ndarray[np.float64_t, ndim=3] sigma, 
        int num_s, 
        int num_p, 
        int num_a_max, 
        np.ndarray[np.int32_t, ndim=2] nums_a
        ):
    
    
    cdef:
        np.ndarray[np.float64_t, ndim=5] out_ = np.zeros(shape=(num_s,num_p,num_a_max,num_s,num_p), dtype=np.float64)
        np.ndarray[np.int32_t, ndim=1] loop_profile = np.zeros(num_p+1, dtype=np.int32)
        double temp_prob 
        int state, player, other, to_state, n
        int flat_index = 0
    
    
    for state in range(num_s):
        for player in range(num_p):
            loop_profile[0:num_p+1] = 0
            while loop_profile[0] == 0:
                
                temp_prob = 1
                for other in range(num_p):
                    if other == player:
                        continue
                    temp_prob *= sigma[state,other,loop_profile[other+1]]
                
                for to_state in range(num_s):
                    out_[ state, player, loop_profile[player+1], to_state, player ] += temp_prob * phi_ravel[flat_index]
                    flat_index += 1
                
                loop_profile[num_p] += 1
                for n in range(num_p):
                    if loop_profile[num_p-n] == nums_a[state,num_p-n-1]:
                        loop_profile[num_p-n-1] += 1
                        loop_profile[num_p-n] = 0
                        flat_index += num_s * (num_a_max - nums_a[state,num_p-n-1]) * num_a_max**n
    
    return out_   






## payoffs u_tilde_{s,i,i',a,a'} (including continuation value) of player i using pure action a in state s, 
## given player i' uses pure action a' and other players use mixed strategy profile sigma[s,p,a]
    ## (the case i'=i is explicitly included)
@cython.boundscheck(False)
@cython.wraparound(False)
def u_tilde_sijab(
        np.ndarray[np.float64_t, ndim=1] u_tilde_ravel, 
        np.ndarray[np.float64_t, ndim=3] sigma, 
        int num_s, 
        int num_p, 
        int num_a_max, 
        np.ndarray[np.int32_t, ndim=2] nums_a
        ):
    
    
    cdef: 
        np.ndarray[np.float64_t, ndim=5] out_ = np.zeros(shape=(num_s,num_p,num_p,num_a_max,num_a_max), dtype=np.float64)
        np.ndarray[np.int32_t, ndim=1] loop_profile = np.zeros(num_p+1, dtype=np.int32)
        ## loop_profile is used to loop over all action profiles
            ## loop_profile[1:num_p+1] gives current action profile, loop_profile[0] in {0,1} indicates whether all action profiles have been explored (1) or not (0)
            ## last element of action profile is iterated first; once that is done, increase second last element by one and set last element to zero again, and so on...
            ## continue until very first element of loop_profile is increased from zero to one, which means all action profiles have been explored
        double temp_prob 
        int state, player1, player2, other, n
        int flat_index = 0
    
    
    for state in range(num_s):
        for player1 in range(num_p):
            for player2 in range(num_p):
                loop_profile[0:num_p+1] = 0
                while loop_profile[0] == 0:
                    
                    temp_prob = 1
                    for other in range(num_p):
                        if other == player1 or other == player2:
                            continue
                        temp_prob *= sigma[state,other,loop_profile[other+1]]
                    
                    out_[ state, player1, player2, loop_profile[player1+1], loop_profile[player2+1] ] += temp_prob * u_tilde_ravel[flat_index]
                    flat_index +=1  
                    
                    loop_profile[num_p] +=1
                    for n in range(num_p):
                        if loop_profile[num_p-n] == nums_a[state,num_p-n-1]:
                            loop_profile[num_p-n-1] += 1
                            loop_profile[num_p-n] = 0
                            flat_index += (num_a_max - nums_a[state,num_p-n-1]) * num_a_max**n
            
                if player2 < num_p - 1:
                    flat_index -= num_a_max**num_p
    
    return out_  






## transition probabilities phi_{s,i,a,s'} of player i using pure action a in state s, 
## given other players use mixed strategy profile sigma[s,p,a]
@cython.boundscheck(False)
@cython.wraparound(False)
def phi_tilde_siat(
        np.ndarray[np.float64_t, ndim=1] phi_ravel, 
        np.ndarray[np.float64_t, ndim=3] sigma, 
        int num_s, 
        int num_p, 
        int num_a_max, 
        np.ndarray[np.int32_t, ndim=2] nums_a
        ):
    
    
    cdef: 
        np.ndarray[np.float64_t, ndim=4] out_ = np.zeros(shape=(num_s,num_p,num_a_max,num_s), dtype=np.float64)
        np.ndarray[np.int32_t, ndim=1] loop_profile = np.zeros(num_p+1, dtype=np.int32)
        ## loop_profile is used to loop over all action profiles
            ## loop_profile[1:num_p+1] gives current action profile, loop_profile[0] in {0,1} indicates whether all action profiles have been explored (1) or not (0)
            ## last element of action profile is iterated first; once that is done, increase second last element by one and set last element to zero again, and so on...
            ## continue until very first element of loop_profile is increased from zero to one, which means all action profiles have been explored
        double temp_prob 
        int state, player, other, to_state, n
        int flat_index = 0
    
    
    for state in range(num_s):
        for player in range(num_p):
            loop_profile[0:num_p+1] = 0
            while loop_profile[0] == 0:
                
                temp_prob = 1
                for other in range(num_p):
                    if other == player:
                        continue
                    temp_prob *= sigma[state,other,loop_profile[other+1]]
                
                for to_state in range(num_s):
                    out_[ state, player, loop_profile[player+1], to_state ] += temp_prob * phi_ravel[flat_index]
                    flat_index += 1
                
                loop_profile[num_p] +=1
                for n in range(num_p):
                    if loop_profile[num_p-n] == nums_a[state,num_p-n-1]:
                        loop_profile[num_p-n-1] += 1
                        loop_profile[num_p-n] = 0
                        flat_index += num_s * (num_a_max - nums_a[state,num_p-n-1]) * num_a_max**n
    
    return out_



## homotopy function H(y)
    ## H(y) = [ H_{strategies}[s,i,a] ]
    ##        [ H_{stateValues}[s,i]  ]
    ## y = [ beta[s,i,a], V[s,i], t ]
        ## within beta and V, respectively, indices are raveled in C-style order 
        ## such that last index varies quickest
@cython.boundscheck(False)
@cython.wraparound(False)
def H(
        np.ndarray[np.float64_t, ndim=1] y, 
        u, 
        phi, 
        int num_s, 
        int num_p, 
        np.ndarray[np.int32_t, ndim=2] nums_a, 
        int num_a_max, 
        int num_a_tot
        ):
    
    
    cdef:
        np.ndarray[np.float64_t, ndim=1] out_ = np.zeros(num_a_tot+num_s*num_p, dtype=np.float64)
        np.ndarray[np.float64_t, ndim=3] beta = np.zeros(shape=(num_s,num_p,num_a_max), dtype=np.float64)
        int state, player, action, a
        int flat_index = 0
    
    ## fill beta
    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state,player]):
                beta[state,player,action] = y[flat_index]
                flat_index += 1
    
    cdef:
        np.ndarray[np.float64_t, ndim=3] sigma = np.exp(beta)
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot:num_a_tot+num_s*num_p].reshape(num_s,num_p)
        double gamma = y[num_a_tot+num_s*num_p]   ## label lambda as gamma in python
        np.ndarray[np.float64_t, ndim=1] u_tilde_ev_ravel = u_tilde(u, V, phi).ravel()
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev = u_tilde_sia(u_tilde_ev_ravel, sigma, num_s, num_p, num_a_max, nums_a)
    
    
    flat_index = 0
    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state,player]):
                
                if action == 0:
                    out_[flat_index] += 1
                    for a in range(nums_a[state,player]):
                        out_[flat_index] -= sigma[state,player,a]
                else:
                    out_[flat_index] = beta[state,player,0] - beta[state,player,action] \
                          - gamma*u_tilde_sia_ev[state,player,0] + gamma*u_tilde_sia_ev[state,player,action] 
                
                flat_index += 1
     
    for state in range(num_s):
        for player in range(num_p):
            out_[flat_index] -= V[state,player]
            for action in range(nums_a[state,player]):
                out_[flat_index] += sigma[state,player,action]*u_tilde_sia_ev[state,player,action] 
            flat_index += 1
    
    return out_
                    



## Jacobian J(y) of homotopy function
    ## J(y) = [  H_{strategies}[s,i,a] / d_beta[s',i',a'],  H_{strategies}[s,i,a] / d_V(s',i'),  H_{strategies}[s,i,a] / d_t  ]
    ##        [   H_{stateValues}[s,i] / d_beta[s',i',a'],   H_{stateValues}[s,i] / d_V(s',i'),   H_{stateValues}[s,i] / d_t  ]
@cython.boundscheck(False)
@cython.wraparound(False)
def J(
        np.ndarray[np.float64_t, ndim=1] y, 
        u, 
        phi, 
        int num_s, 
        int num_p, 
        np.ndarray[np.int32_t, ndim=2] nums_a, 
        int num_a_max, 
        int num_a_tot
        ):
    
    
    cdef:
        np.ndarray[np.float64_t, ndim=2] out_ = np.zeros(shape=(num_a_tot+num_s*num_p,num_a_tot+num_s*num_p+1), dtype=np.float64)
        np.ndarray[np.float64_t, ndim=3] beta = np.zeros(shape=(num_s,num_p,num_a_max), dtype=np.float64)
        int state, player, action
        int flat_index = 0
    
    ## fill beta
    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state,player]):
                beta[state,player,action] = y[flat_index]
                flat_index += 1
    
    cdef:
        np.ndarray[np.float64_t, ndim=3] sigma = np.exp(beta)
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot:num_a_tot+num_s*num_p].reshape(num_s,num_p)
        double gamma = y[num_a_tot+num_s*num_p]   ## label lambda as gamma in python
        np.ndarray[np.float64_t, ndim=1] u_tilde_ev_ravel = u_tilde(u, V, phi).ravel()
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev = u_tilde_sia(u_tilde_ev_ravel, sigma, num_s, num_p, num_a_max, nums_a)
        np.ndarray[np.float64_t, ndim=5] u_tilde_sia_partial_beta_ev = u_tilde_sia_partial_beta(u_tilde_ev_ravel, sigma, num_s, num_p, num_a_max, nums_a)
        np.ndarray[np.float64_t, ndim=5] u_tilde_sia_partial_V_ev = u_tilde_sia_partial_V(phi.ravel(), sigma, num_s, num_p, num_a_max, nums_a)
        int row_state, row_player, row_action
        int col_state, col_player, col_action
        int row_index, col_index, col_index_init
    
    
    ## first block: rows with H_{strategies}[s,i,a]
    row_index = 0
    col_index_init = 0
    for row_state in range(num_s):
        for row_player in range(num_p):
            for row_action in range(nums_a[row_state,row_player]):
                
                ## derivatives w.r.t. beta[s',i',a']
                ## entries with s' != s are 0, thus no looping over s'
                col_index = col_index_init
                for col_player in range(num_p):
                    for col_action in range(nums_a[row_state,col_player]):
                        
                        ## diagonal sub-blocks: derivatives w.r.t. beta[s,i,a'] (own actions in same state)
                        if row_player == col_player:
                            if row_action == 0:
                                out_[row_index, col_index] = -sigma[row_state,col_player,col_action]
                            else:
                                if col_action == 0:
                                    out_[row_index, col_index] = 1
                                elif row_action == col_action:
                                    out_[row_index, col_index] = -1
                        
                        ## off-diagonal sub-blocks: derivatives w.r.t. beta[s,i',a'] (other players' actions in same state)
                        else:
                            ## row_action == 0 -> entry = 0
                            if row_action != 0:
                                out_[row_index, col_index] = gamma * ( u_tilde_sia_partial_beta_ev[row_state,row_player,row_action,col_player,col_action]
                                                                       - u_tilde_sia_partial_beta_ev[row_state,row_player,       0,col_player,col_action] )
                        
                        col_index += 1
                
                ## derivatives w.r.t. V[s',i']
                col_index = num_a_tot
                for col_state in range(num_s):
                    for col_player in range(num_p):
                        if row_action != 0:
                            out_[row_index, col_index] = gamma * ( u_tilde_sia_partial_V_ev[row_state,row_player,row_action,col_state,col_player]
                                                                   - u_tilde_sia_partial_V_ev[row_state,row_player,       0,col_state,col_player] )
                        col_index += 1
                
                ## derivative w.r.t. lambda
                ## row_action == 0 -> entry = 0
                if row_action != 0:
                    out_[row_index, col_index] = u_tilde_sia_ev[row_state,row_player,row_action] - u_tilde_sia_ev[row_state,row_player,0]
                
                row_index += 1
        
        for row_player in range(num_p):
            col_index_init += nums_a[row_state,row_player]
    
    
    ## second block: rows with H_{stateValues}[s,i]
    row_index = num_a_tot
    col_index_init = 0
    for row_state in range(num_s):
        for row_player in range(num_p):
            
            ## derivatives w.r.t. beta[s',i',a']
            ## entries with s' != s are 0, thus no looping over s'
            col_index = col_index_init
            
            for col_player in range(num_p):
                for col_action in range(nums_a[row_state,col_player]):
                    
                    if row_player == col_player:
                        out_[row_index,col_index] += sigma[row_state,row_player,col_action] * u_tilde_sia_ev[row_state,row_player,col_action]
                    else:
                        for row_action in range(nums_a[row_state,row_player]):
                            out_[row_index,col_index] += sigma[row_state,row_player,row_action] * u_tilde_sia_partial_beta_ev[row_state,row_player,row_action,col_player,col_action]
                    
                    col_index += 1
            
            ## derivatives w.r.t. V[s',i']
            col_index = num_a_tot
            for col_state in range(num_s):
                for col_player in range(num_p):
                    
                    if col_state == row_state and col_player == row_player:
                        out_[row_index,col_index] -=1
                    for row_action in range(nums_a[row_state,row_player]):
                        out_[row_index,col_index] += sigma[row_state,row_player,row_action] * u_tilde_sia_partial_V_ev[row_state,row_player,row_action,col_state,col_player]
                    
                    col_index += 1
                            
            ## derivative w.r.t. lambda := 0
            
            row_index += 1
        
        for row_player in range(num_p):
            col_index_init += nums_a[row_state,row_player]
    
    return out_
