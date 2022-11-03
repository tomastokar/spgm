from src.classes import *

p_ab = CPDTable('A', [[.1, .7], [.9, .3]], ['B'])
p_bc = CPDTable('B', [[.6, .5], [.4, .5]], ['C'])
p_c  = CPDTable('C', [[.2], [.8]])
edges = [('C', 'B'), ('B', 'A')]
model = BNet(edges)
model.add_cpd(p_ab)
model.add_cpd(p_bc)
model.add_cpd(p_c)
# model = model.eliminate_variable('C')
# model = model.eliminate_variable('B')
runner = QRunner(model)
model = runner.run_query('A', 'C')