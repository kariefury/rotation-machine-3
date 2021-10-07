# Makes a graphiz plot of a hierarchy.
import graphviz as gv

g1 = gv.Digraph(format='png')
g1.node('0',label="A",shape='circle')
g1.node('1', label="B",shape='circle')
g1.node('2', label="C",shape='circle')
g1.edge('0', '1',label='x = 1')
g1.edge('1', '2',label='x = 1')
g1.edge('2', '0',label='x = 1')
g1.edge('0', '2',label='x = 0')
g1.edge('2', '1',label='x = 0')
g1.edge('1', '0',label='x = 0')
g1.edge('1','1',label='p = 0.2')
g1.edge('2','2',label='p = 0.2')
g1.edge('0','0',label='p = 0.2')

filename = g1.render(filename='fig/driven_state_machine')