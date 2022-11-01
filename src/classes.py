import itertools
import numpy as np

class CPDTable:
    def __init__(self, variable, values, evidence = []):
        self.variable = variable
        self.evidence = evidence
        self.k = 1 + len(evidence)
        self.values = np.array(values)
        self.values = self.values.reshape([2] * self.k)
    
    def _normalize(self):
        self.values /= self.values.sum(axis = 0)
    
    def marginalize(self, variable):        
        i = self.evidence.index(variable)
        values = self.values.sum(axis = i + 1)
        evidence = [e for e in self.evidence if e != variable]
        cpd_table = CPDTable(self.variable, values, evidence)
        return cpd_table

    def reduce(self, variable, value):           
        i = self.evidence.index(variable)
        l = self.legend[:,i]
        idx = np.where(l == value)[0]
        values = self.values[:,i + ]
        evidence = [e for e in self.evidence if e != variable]
        cpd_table = CPDTable(self.variable, values, evidence)
        return cpd_table
    
        
class BNet:
    def __init__(self, edges):
        self.edges = edges
        self.nodes = set([node for edge in edges for node in edge])
        self.in_degree, self.out_degree = self._get_degree()
        self.cpd_tables = []
        
    def _get_degree(self):
        in_degree = {n : 0 for n in self.nodes}
        out_degree = {n : 0 for n in self.nodes}
        for edge in self.edges:
            in_degree[edge[0]] += 1
            out_degree[edge[1]] += 1
        return in_degree, out_degree
    
    def add_cpd(self, cpd_table):
        self.cpd_tables.append(cpd_table)
        
    def elminate_variable(self, variable):
        cpds = []
        for cpd in self.cpd_tables:
            if variable in cpd.evidence:
                
    
    def eval_query(self, variable, evidence = []):
        keep_nodes = set(variable).union(evidence)
        elim_nodes = [node for node in self.nodes if node not in keep_nodes]
        
        if len(elim_nodes) > 0:
            elimination_order = sorted(elim_nodes, key=self.in_degee.get)
        
        