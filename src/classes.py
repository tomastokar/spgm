
import functools
import numpy as np

def multiply_pds(pd_1, pd_2):
    vals_1 = pd_1.values
    vals_2 = pd_2.values
    vars_1 = pd_1.variables
    vars_2 = pd_2.variables
    new_vars = set(vars_1 + vars_2)
    var_dict = {var : i for i, var in enumerate(new_vars)}
    new_vals = np.einsum(
        vals_1,
        [var_dict[var] for var in vars_1],
        vals_2,
        [var_dict[var] for var in vars_2],
        range(len(new_vars))
    )
    jpd_table = JPDTable(new_vars, new_vals)
    return jpd_table
    

class PDTable:
    def reduce(self, variable, value):           
        i = self.variables.index(variable)        
        values = np.take(self.values, value, i)
        return values

    def marginalize(self, variable):        
        i = self.variables.index(variable)
        values = self.values.sum(axis = i)
        return values
    

class CPDTable(PDTable):
    def __init__(self, target, values, evidence = []):
        self.target = target
        self.evidence = evidence
        self.variables = [target] + evidence
        k = len(self.variables)
        self.values = np.array(values).reshape([2] * k)
        # self._normalize()
    
    def _normalize(self):
        self.values /= self.values.sum(axis = 0)
    
    def marginalize(self, variable):                
        values = super().marginalize(variable)
        evidence = [e for e in self.evidence if e != variable]
        cpd_table = CPDTable(self.target, values, evidence)
        return cpd_table

    def reduce(self, variable, value):               
        values = super().reduce(variable, value)
        evidence = [e for e in self.evidence if e != variable]
        cpd_table = CPDTable(self.target, values, evidence)
        return cpd_table
    
    def eliminate_evidence(self, cpd_table):
        var = cpd_table.target
        val = cpd_table.values
        i = self.evidence.index(var) + 1
        values = np.tensordot(self.values, val, axes = (i, 0))        
        evidence = [e for e in self.evidence if e != var]
        evidence += cpd_table.evidence
        cpd_table = CPDTable(self.target, values, evidence)
        return cpd_table


class JPDTable(PDTable):
    def __init__(self, variables, values):
        self.variables = list(set(variables))
        k = len(self.variables)
        self.values = np.array(values)
        self.values = self.values.reshape([2] * k)
        # self._normalize()
    
    def _normalize(self):
        self.values /= self.values.sum()
    
    def marginalize(self, variable):                
        values = super().marginalize(variable)
        variables = [e for e in self.variables if e != variable]
        jpd_table = JPDTable(variables, values)
        return jpd_table

    def reduce(self, variable, value):               
        values = super().reduce(variable, value)
        variables = [e for e in self.variables if e != variable]
        jpd_table = JPDTable(variables, values)
        return jpd_table    


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
        self.cpd_tables[cpd_table.target] = cpd_table
        
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

        cpd_variable = self.cpd_tables[variable]
        for k, cpd in self.cpd_tables.items():
            if variable in cpd.evidence:
                cpd_ = cpd.eliminate_evidence(cpd_variable)
                model.add_cpd(cpd_)
            elif variable != k:
                model.add_cpd(cpd)        
        return model
                
        
class QRunner:
    def __init__(self, model):
        self.model = model
    
    def calc_minfill_cost(self, edges, node):
        degree = len([edge for edge in edges in node in edges])
        return degree * (degree - 1) / 2
            
    def resolve_order(self, variables):
        if len(variables) > 1:         
            edges = self.model.edges
            ordering = []   
            while variables:
                scores = {v : self.calc_minfill_cost(edges, v) for v in variables}        
                pick = min(scores, key = scores.get())
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
    
    def resolve_query(self, variable, evidence = None, model = None):
        if model is None:
            model = self.model
        # Calculate joint distribution
        jpd = functools.reduce(
            lambda x, y: multiply_pds(x, y), 
            list(model.cpd_tables.values())
        )
        # Marginalize on evidence variables                
        if evidence is not None:
            jpd_ = jpd.marginalize(variable)
            jpd_.values = 1. / jpd_.values
            jpd = multiply_pds(jpd, jpd_)        
        return jpd
                                    
    def run_query(self, variable, evidence = None):
        # Select variable to eliminate
        query_nodes = set([variable, evidence])
        elim_nodes = [n for n in self.model.nodes if n not in query_nodes]        
        # Eliminate variables
        model = self.eliminate_variables(elim_nodes)            
        # Resolve query
        pd_table = self.resolve_query(variable, evidence, model)                
        return pd_table
