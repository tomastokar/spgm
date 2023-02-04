import functools
import numpy as np
from collections import Counter, defaultdict


class CPDTable():
    def __init__(self, target, values, evidence = []):
        assert set(target).intersection(evidence) == set()
        self.target = target
        self.evidence = evidence
        self.variables = target + evidence
        self.values = np.array(values)
        self.values.shape = ([2] * len(self.variables))
    
    def marginalize(self, variables):        
        cpd = self.copy()
        for variable in variables:
            assert variable in cpd.target
            i = cpd.variables.index(variable)
            values = cpd.values.sum(axis = i)
            target = [t for t in cpd.target if t != variable]
            evidence = [v for v in cpd.evidence if v != variable]
            cpd = CPDTable(target, values, evidence)
        return cpd

    def reduce(self, state):
        cpd = self.copy()
        for var, val in state.items():
            if var in cpd.evidence:
                i = cpd.variables.index(var)
                values = np.take(cpd.values, val, i)            
                evidence = [v for v in cpd.evidence if v != var]
                cpd = CPDTable(cpd.target, values, evidence)
        return cpd
    
    def reorder(self, target, evidence):        
        idx = {v:i for i,v in enumerate(self.variables)}
        order = [idx[v] for v in target + evidence]
        values = np.transpose(self.values, order)
        cpd = CPDTable(target, values, evidence)
        return cpd
    
    def recondition(self, variables):    
        cpd = self.copy()
        for variable in variables:
            assert variable in cpd.target
            # Select variable to marginalize on
            to_marginalize = [t for t in cpd.target if t != variable]        
            # Marginalize over the selected variables
            denominator = cpd.marginalize(to_marginalize)
            # Invert values
            denominator.values = 1./denominator.values
            # Get value of a new conditional 
            cpd = cpd.multiply(denominator)
            # Move variables from target to evidence
            target = [v for v in cpd.target if v != variable]
            evidence = cpd.evidence + [variable]
            # Reorder axes
            cpd = cpd.reorder(target, evidence)
        return cpd
         
    def multiply(self, cpd):
        target = set(self.target + cpd.target)
        evidence = set(self.evidence + cpd.evidence) - target     
        var_dict = {}
        for i, var in enumerate(target):
            var_dict[var] = i
        for j, var in enumerate(evidence):
            var_dict[var] = j + len(target)                
        values = np.einsum(
            self.values,
            [var_dict[var] for var in self.variables],
            cpd.values,
            [var_dict[var] for var in cpd.variables],
            range(len(var_dict))
        )
        cpd = CPDTable(list(target), values, list(evidence))
        return cpd
        
    def select(self, state):
        variables = self.variables
        values = self.values
        for var, val in state.items():
            if var in variables:
                i = variables.index(var)
                values = np.take(values, val, i)            
                variables = [v for v in variables if v != var]                
        return values, variables        
        
    def copy(self):
        return CPDTable(self.target, self.values, self.evidence)


class BNet:
    def __init__(self, edges):
        self.edges = edges
        self.nodes = set([node for edge in edges for node in edge])                
        self.degree = self.calc_degree()
        self.parents = self.get_parents()
        self.children = self.get_children()    
        self.neighbors = self.get_neighbors()
        self.blankets = self.get_blankets()
        self.cpd_tables = []        
        
    def calc_degree(self):
        degree = Counter([n for edge in self.edges for n in edge])
        return degree


    def get_parents(self):
        parents = defaultdict(set)
        for u, v in self.edges:    
            parents[v] = parents[v].union([u])
        return parents

            
    def get_children(self):
        children = defaultdict(set)
        for u, v in self.edges:    
            children[u] = children[u].union([v])
        return children
    
    def get_neighbors(self):
        neighbors = defaultdict(set)
        for n in self.nodes:
            neighbors[n] = self.children[n].union(self.parents[n])
        return neighbors

    def get_blankets(self):       
        blankets = defaultdict(set)
        for n in self.nodes:
            blankets[n] = self.neighbors[n]
            blankets[n] = blankets[n].union([p for ch in self.children[n] for p in self.parents[ch] if p != n])
        return blankets        
                                                                                    
    def add_cpd(self, cpd_table):        
        self.cpd_tables.append(cpd_table)
        
    def reroute_edges(self, node):    
        edges = [edge for edge in self.edges if node not in edge]            
        edges += [[p, ch] for p in self.parents[node] for ch in self.children[node]]   
         
        return edges    

    def create_factor(self, variable):
        # Select cpds containing the variable
        cpds = [cpd for cpd in self.cpd_tables if variable in cpd.variables]
        # Calculate new cpd table
        cpd = functools.reduce(lambda x, y: x.multiply(y), cpds)
        # Marginalize along variable
        cpd = cpd.marginalize([variable])
                
        return cpd    
        
    def eliminate_variable(self, variable):        
        # Get new edges
        edges = self.reroute_edges(variable)                                
        # Select cpds for new model
        cpds = [cpd for cpd in self.cpd_tables if variable not in cpd.variables]        
        # Create a new factor
        cpds.append(self.create_factor(variable))
        # Create new model
        model = BNet(edges)
        for cpd in cpds:
            model.add_cpd(cpd)
            
        return model
                

class MyVariableElimination:
    def __init__(self, model):
        self.model = model
        
    def calc_minfill_scores(self, variables):                
        scores = {}
        for v in variables:
            d = self.model.degree[v]
            scores[v] = d * (d - 1) / 2 
        return scores        
                
    def resolve_order(self, variables):
        if len(variables) > 1:         
            edges = self.model.edges            
            ordering = []   
            while variables:
                scores = self.calc_minfill_scores(variables)
                pick = min(scores, key = scores.get)
                ordering.append(pick)  
                variables.remove(pick)
                edges = [edge for edge in edges if pick not in edge]                         
        else:
            ordering = variables            
        print('Variables to eliminate in order: {}'.format(', '.join(ordering)))        
        return ordering
        
    def eliminate_variables(self, variables):
        # Resolve order of elimination        
        variables = self.resolve_order(variables)                    
        # Variable elimination
        model = self.model
        for variable in variables:
            model = model.eliminate_variable(variable)        
        return model
    
    def answer_query(self, model, evidence = {}):
        # Calculate joint distribution
        cpd = functools.reduce(lambda x, y: x.multiply(y), model.cpd_tables)
        # Condition of evidence
        if evidence:
            cpd = cpd.recondition(list(evidence.keys()))
            cpd = cpd.reduce(evidence)            
        return cpd
                                    
    def run_query(self, variables, evidence = {}):
        # Query variables
        query_nodes = set(variables).union(evidence)
        # Select variable to eliminate        
        elim_nodes = [n for n in self.model.nodes if n not in query_nodes]        
        # Eliminate variables
        model = self.eliminate_variables(elim_nodes)            
        # Answer query
        cpd_tables = self.answer_query(model, evidence)       
        return cpd_tables
    

class MyGibbsSampler:
    def __init__(self, model, no_iters = 1e+6, thinning = 100, burn_in = 1e+3, conf_level = 0.95):
        self.model = model
        self.burn_in = burn_in        
        self.no_iters = no_iters
        self.thinning = thinning
        self.conf_level = conf_level       
        self.get_kernels()          

    # def reduce_cpds(self, evidence):
    #     # Reduce cpds given the evidence
    #     cpds = [cpd.reduce(evidence) for cpd in self.model.cpd_tables]
        
    #     # Ged rid of cpds of the evidence variables
    #     for e in evidence.keys():
    #         cpds = [cpd for cpd in cpds if e not in cpd.target]
        
    #     return cpds

    # def get_trans_probas(self, evidence):

    #     # Calculate transition probabilities
    #     trans_probas = {}
    #     for node in self.model.nodes:            
    #         if node not in evidence.keys():
    #             # Select cpd of the node
    #             cpds = [cpd for cpd in self.model.cpd_tables if node in cpd.target]
                
    #             # Add cpds associated with the node's children                 
    #             children = self.model.children[node]
    #             for ch in children:
    #                 cpds += [cpd for cpd in self.model.cpd_tables if ch in cpd.target]  
                
    #             # Calculate joint distribution                            
    #             joint_proba = functools.reduce(lambda x, y: x.multiply(y), cpds)            
                
    #             # Select variables to condition on
    #             condition_vars = [t for t in joint_proba.target if t != node]
                
    #             # Convert joint to conditional
    #             trans_probas[node] = joint_proba.recondition(condition_vars)
                                
    #     return trans_probas
    
    def get_kernels(self):
        self.kernels = {}
        for n in self.model.nodes:
            self.kernels[n] = [cpd for cpd in self.model.cpd_tables if n in cpd.target]
            for ch in self.model.children[n]:
                self.kernels[n] += [cpd for cpd in self.model.cpd_tables if ch in cpd.target]
                
    def calc_trans_proba(self, var):
        # Reduce current state to non-var variables
        state = {v : s for v, s in self.state.items() if v != var}
                
        # Take cpds from variable kernel
        cpds = self.kernels[var]
        
        # Select probas given the state
        log_probas = []
        for cpd in cpds:            
            p, vars = cpd.select(state)
            p = p.flatten()
            log_probas.append(np.log(p))
            # print(var, vars, p, np.log(p))
        
        # Calculate joint proba
        joint_proba = functools.reduce(lambda x, y: x + y, log_probas)
        # print(joint_proba)
        
        joint_proba = np.exp(joint_proba)
        # print(joint_proba)
                            
        # Normalize joint proba 
        trans_proba = joint_proba / joint_proba.sum()
        
        # print(trans_proba)
        
        return trans_proba

    def sample_state(self, proba):        
        r = np.random.rand()
        w = np.cumsum(proba)
        return np.searchsorted(w, r, side='right')
    
    def initialize(self):
        self.state = {}
        for n in self.model.nodes:            
            if n in self.evidence.keys():
                self.state[n] = self.evidence[n]
            else:
                self.state[n] = self.sample_state([0.5])
            
    def update_variable(self, var):
        # Update state of the variable
        p = self.calc_trans_proba(var)
        self.state[var] = self.sample_state(p)
                
    def update_state(self):    
        # Update current state    
        for v in self.state.keys():
            if v not in self.evidence:
                self.update_variable(v)
                     
    def record_state(self):
        # Add variable states to observations
        for v in self.variables:
            self.observations[v].append(self.state[v])
                 
    def gibbs_sampling(self):          
        # Init container for observations
        self.observations = defaultdict(list)                
        # Init state
        self.initialize()                       
        # Run burn-in
        for _ in range(int(self.burn_in)):
            self.update_state()            
        # Run markov chain        
        for i in range(int(self.no_iters)):
            self.update_state()                        
            # Record state
            if i % self.thinning == 0: 
                self.record_state()
    
    def run_query(self, variables, evidence = {}):
        self.variables = variables
        self.evidence = evidence
        self.gibbs_sampling()

    def calc_stats(self, max_obs):        
        # Reduce observations to max observations
        # to be used to calculate statistics
        # (this is useful to demonstrate convergence)
        observations = {var : x[:max_obs] for var, x in self.observations.items()}
            
        stats = {}
        for n, x in observations.items():
            mu = np.mean(x)
            ci = self.conf_level * np.std(x) / np.sqrt(len(x))
            stats[n] = [mu, ci]
        return stats
        
    def get_results(self, max_obs = None):
        if max_obs is None:
            max_obs = len(self.observations)
        self.stats = self.calc_stats(max_obs)         
        return self.stats
