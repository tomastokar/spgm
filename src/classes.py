
import functools
import numpy as np

    
class CPDTable():
    def __init__(self, target, values, evidence = []):
        self.target = target
        self.evidence = evidence
        self.variables = target + evidence
        self.values = np.array(values)
        self.values.shape = ([2] * len(self.variables))
    
    def marginalize(self, variable):                
        i = self.variables.index(variable)
        values = self.values.sum(axis = i)
        target = [t for t in self.target if t != variable]
        evidence = [v for v in self.evidence if v != variable]
        cpd_table = CPDTable(target, values, evidence)
        return cpd_table

    def reduce(self, variable, value):               
        i = self.variables.index(variable)        
        values = np.take(self.values, value, i)
        target = [t for t in self.target if t != variable]
        evidence = [v for v in self.evidence if v != variable]
        cpd_table = CPDTable(target, values, evidence)
        return cpd_table
    
    def multiply(self, cpd_table):
        target = set(self.target + cpd_table.target)
        evidence = set(self.evidence + cpd_table.evidence) - target     
        var_dict = {}
        for i, var in enumerate(target):
            var_dict[var] = i
        for j, var in enumerate(evidence):
            var_dict[var] = j + len(target)                
        new_vals = np.einsum(
            self.values,
            [var_dict[var] for var in self.variables],
            cpd_table.values,
            [var_dict[var] for var in cpd_table.variables],
            range(len(var_dict))
        )
        cpd_table = CPDTable(list(target), new_vals, list(evidence))
        return cpd_table



class BNet:
    def __init__(self, edges):
        self.edges = edges
        self.nodes = set([node for edge in edges for node in edge])
        self.in_degree, self.out_degree = self._get_degree()
        self.cpd_tables = {}
        
    def _get_degree(self):
        in_degree = {n : 0 for n in self.nodes}
        out_degree = {n : 0 for n in self.nodes}
        for edge in self.edges:
            in_degree[edge[0]] += 1
            out_degree[edge[1]] += 1
        return in_degree, out_degree
    
    def add_cpd(self, cpd_table):
        assert len(cpd_table.target) == 1
        self.cpd_tables[cpd_table.target[0]] = cpd_table
        
    def reroute_edges(self, variable):
        edges = []
        fr = []; to = []   
        for edge in self.edges:
            if edge[0] == variable:
                to.append(edge[1])                
            elif edge[1] == variable:
                fr.append(edge[0])                
            else:
                edges.append(edge)                   
        if len(fr) > 0:
            edges += [[n,m] for n in fr for m in to]
        return edges                     
        
    def eliminate_variable(self, variable):        
        edges = self.reroute_edges(variable)
        model = BNet(edges)        

        cpd_table = self.cpd_tables[variable]
        for k, cpd in self.cpd_tables.items():
            if variable in cpd.evidence:
                cpd = cpd.multiply(cpd_table)
                cpd = cpd.marginalize(variable)
                model.add_cpd(cpd)
            elif k != variable:
                model.add_cpd(cpd)        
        return model
                
        
class QRunner:
    def __init__(self, model):
        self.model = model
    
    def calc_minfill_cost(self, edges, node):
        degree = len([edge for edge in edges if node in edge])
        return degree * (degree - 1) / 2
            
    def resolve_order(self, variables):
        if len(variables) > 1:         
            edges = self.model.edges            
            ordering = []   
            while variables:
                scores = {v : self.calc_minfill_cost(edges, v) for v in variables}                     
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
    
    def resolve_query(self, model, variable, evidence = None):
        # Calculate joint distribution
        cpd = functools.reduce(
            lambda x, y: x.multiply(y), 
            list(model.cpd_tables.values())
        )
        # Marginalize on variables
        if evidence is not None:
            cpd_ = cpd.marginalize(variable)
            cpd_.values = 1. / cpd_.values
            # cpd = cpd.multiply(cpd_)        
        return cpd, cpd_
                                    
    def run_query(self, variable, evidence = None):
        # Select variable to eliminate
        query_nodes = set([variable, evidence])
        elim_nodes = [n for n in self.model.nodes if n not in query_nodes]        
        # Eliminate variables
        model = self.eliminate_variables(elim_nodes)            
        # Resolve query
        cpd_tables = self.resolve_query(model, variable, evidence)                
        return cpd_tables
