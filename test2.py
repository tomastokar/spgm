from src.classes import *

edges = [
    ("Burglary", "Alarm"),
    ("Earthquake", "Alarm"),
    ("Alarm", "JohnCalls"),
    ("Alarm", "MaryCalls"),
]
model = BNet(edges)

cpd_burglary = CPDTable(
    target = ["Burglary"], 
    values=[[0.999], [0.001]]
)

cpd_earthquake = CPDTable(
    target=["Earthquake"], 
    values=[[0.998], [0.002]]
)

cpd_alarm = CPDTable(
    target=["Alarm"],
    values=[
        [0.999, 0.71, 0.06, 0.05], 
        [0.001, 0.29, 0.94, 0.95]
    ],
    evidence=["Burglary", "Earthquake"]
)
cpd_johncalls = CPDTable(
    target=["JohnCalls"],
    values=[
        [0.95, 0.1], 
        [0.05, 0.9]
    ],
    evidence=["Alarm"]    
)

cpd_marycalls = CPDTable(
    target=["MaryCalls"],    
    values=[
        [0.1, 0.7], 
        [0.9, 0.3]
    ],
    evidence=["Alarm"],
)

model.add_cpd(cpd_burglary)
model.add_cpd(cpd_earthquake)
model.add_cpd(cpd_alarm)
model.add_cpd(cpd_johncalls)
model.add_cpd(cpd_marycalls)
# model = model.eliminate_variable('C')
# model = model.eliminate_variable('B')
runner = QRunner(model)
res = runner.run_query('MaryCalls', evidence = 'JohnCalls')