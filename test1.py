from src.classes import *

p_ab = CPDTable(
    target = ['A'], 
    values = [
        [.1, .7], 
        [.9, .3]
    ], 
    evidence = ['B']
)

p_bc = CPDTable(
    target = ['B'], 
    values = [
        [.6, .5], 
        [.4, .5]
    ], 
    evidence = ['C']
)

p_c  = CPDTable(
    target = ['C'], 
    values = [
        [.2], [.8]
    ]
)
edges = [
    ('C', 'B'), 
    ('B', 'A')
]
model = BNet(edges)

model.add_cpd(p_ab)
model.add_cpd(p_bc)
model.add_cpd(p_c)

# model = model.eliminate_variable('C')
# model = model.eliminate_variable('B')

runner = QRunner(model)
res = runner.run_query('A', 'C')