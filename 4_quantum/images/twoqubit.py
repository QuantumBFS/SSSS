import matplotlib.pyplot as plt
from viznet import DynamicShow, QuantumCircuit, NodeBrush
from viznet import parsecircuit as _
import pdb
from viznet import setting
setting.node_setting["lw"] = 2
setting.node_setting["inner_lw"] = 2
import numpy as np

def twoqubit():
    INIT = NodeBrush("tn.tri", size=0.36, color="none", rotate=np.pi/6)
    PIN = NodeBrush("pin")

    handler = QuantumCircuit(num_bit=2, y0=2.)
    handler.x -= 0.5

    itheta = [0]
    def count():
        itheta[0] += 1
        return itheta[0]

    def reset():
        handler.edge.lw = 0
        handler.gate(INIT, 0, "0", fontsize=12)
        handler.edge.lw = 2

    def rxgate(i):
        handler.gate(_.WIDE, i, r'$R_x^{%d}$'%count(), fontsize=14)

    def rzgate(i):
        handler.gate(_.WIDE, i, r'$R_z^{%d}$'%count(), fontsize=14)

    def cphase():
        handler.gate((_.C, _.GATE), (0, 1), ["", r"$\theta^{%d}$"%count()], fontsize=12)

    with DynamicShow((12, 2), 'twoqubit.png') as ds:
        depth = 2
        for i in range(3):
            reset()
            if i==0:
                handler.gate(INIT, 1, '0', fontsize=12)
                handler.gate(INIT, 0, '0', fontsize=12)
            handler.x += 1
            #for di in range(depth):
            with handler.block(slice(0,1), pad_x=0.2, pad_y=0.1) as b:
                rxgate(0)
                rxgate(1)
                handler.x += 1.0
                rzgate(0)
                rzgate(1)

                handler.x += 1.0
                cphase()

            b[0].text(r"$\times d$", "top")

            handler.x += 1
            handler.gate(_.MEASURE, 0)
            if i!=2:
                handler.x += 0.7
                handler.gate(_.GATE, 0, r'$\circlearrowleft$', fontsize=16)
                handler.x += 0.5
            else:
                handler.gate(_.MEASURE, 1)

def fourqubit():
    nbit = 4
    INIT = NodeBrush("tn.tri", size=0.36, color="none", rotate=np.pi/6)
    PIN = NodeBrush("pin")

    handler = QuantumCircuit(num_bit=nbit, y0=2.)
    handler.x -= 0.5

    itheta = [0]
    def count():
        itheta[0] += 1
        return itheta[0]

    def reset():
        handler.edge.lw = 0
        handler.gate(INIT, 0, "0", fontsize=12)
        handler.edge.lw = 2

    def rxgate(i):
        handler.gate(_.WIDE, i, r'$R_x^{%d}$'%count(), fontsize=14)

    def rzgate(i):
        handler.gate(_.WIDE, i, r'$R_z^{%d}$'%count(), fontsize=14)

    def cphase(i,j):
        handler.gate((_.C, _.GATE), (i,j), ["", r"$\theta^{%d}$"%count()], fontsize=12)


    with DynamicShow((8, 3), 'fourqubit.png') as ds:
        handler.edge.lw = 2
        for i in range(nbit):
            handler.gate(INIT, i, '0', fontsize=12)

        handler.x += 1

        for i in range(nbit-1):
            with handler.block(slice(i,nbit-1), pad_x=0.2, pad_y=0.1) as b:
                rxgate(i)
                rxgate(nbit-1)
                handler.x += 1.0
                rzgate(i)
                rzgate(nbit-1)

                handler.x += 1.0
                cphase(i, nbit-1)
            b[0].text(r"$\times d$", "top")
            handler.x += 1.2

        for i in range(nbit):
            handler.gate(_.MEASURE, i)

if __name__ == '__main__':
    twoqubit()
    fourqubit()
