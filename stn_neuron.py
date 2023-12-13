import sys, os
from neuron import h
import neuron as nrn
from neuron.units import ms, mV
# import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

path = os.getcwd()
nrn.load_mechanisms(path+'/sth')
h.load_file('stdrun.hoc')
PI = np.pi
fiberD=2
paralength1=3  
nodelength=1
space_p1=0.002  
space_p2=0.004
space_i=0.004
rhoa = 7e5
nodeD = 1.4
axonD = 1.6
paraD1 = 1.4
paraD2 = 1.6
mycm=0.1
mygm = 0.001
nl=30
Rpn0=(rhoa*.01)/(PI*((((nodeD/2)+space_p1)**2)-((nodeD/2)**2)))
Rpn1=(rhoa*.01)/(PI*((((paraD1/2)+space_p1)**2)-((paraD1/2)**2)))
Rpn2=(rhoa*.01)/(PI*((((paraD2/2)+space_p2)**2)-((paraD2/2)**2)))
Rpx=(rhoa*.01)/(PI*((((axonD/2)+space_i)**2)-((axonD/2)**2)))


# Read the pool
with open('MatingPool.pickle','rb') as p_file:
        MatingPool = pickle.load(p_file)
index_min = MatingPool[1].index(min(MatingPool[1]))
f_min = MatingPool[0][index_min]


def read_dat(file_name):
    df = pd.read_csv('sth/sth-data/' + file_name, header=None, sep=' ')
    df = df.sort_values(by=[0])
    df = df.reset_index(drop=True)
    return df


def read_channel_distribution(g_name, c_name):
    df = pd.read_csv('sth/sth-data/cell-'+g_name+'_'+c_name, header=None, sep=' ')
    df.columns = ['sec_name', 'sec_ref', 'seg', 'val']
    return df


def find_dat(i, df):
    children = [df[1][i]-1, df[2][i]-1]
    diam = df[3][i]
    L = df[4][i]
    nseg = df[5][i]
    return [children, diam, L, nseg]


def insert_channels(sec):
    sec.insert('STh')
    sec.insert('Na')
    sec.insert('NaL')
    sec.insert('KDR')
    sec.insert('Kv31')
    sec.insert('Ih')
    sec.insert('Cacum')
    sec.insert('sKCa')
    sec.insert('CaT')
    sec.insert('HVA')
    sec.insert('extracellular')
    for i in range(1):
        sec.xraxial[i] = 1e9
        sec.xg[i] = 1e9
        sec.xc[i] = 0


def set_aCSF(i):
    if i == 3:
        # print("Setting in vitro parameters based on Beurrier et al (1999)")
        h.nai0_na_ion = 15
        h.nao0_na_ion = 150   
        h.ki0_k_ion = 140
        h.ko0_k_ion = 3.6   
        h.cai0_ca_ion = 1e-04
        h.cao0_ca_ion = 2.4    
#         h.cli0_cl_ion = 4
#         h.clo0_cl_ion = 135
    if i == 4:
        # print("Setting in vitro parameters based on Bevan & Wilson (1999)")
        h.nai0_na_ion = 15             # This is different in Miocinovic model
        h.nao0_na_ion = 128.5          # This is different in Miocinovic model
        h.ki0_k_ion = 140
        h.ko0_k_ion = 2.5
        h.cai0_ca_ion = 1e-04
        h.cao0_ca_ion = 2.0
#         h.cli0_cl_ion = 4
#         h.clo0_cl_ion = 132.5
    if i == 0:
        # print("WARNING: Using NEURON defaults for in vitro parameters")
        h.nai0_na_ion = 10
        h.nao0_na_ion = 140
        h.ki0_k_ion = 54
        h.ko0_k_ion = 2.5
        h.cai0_ca_ion = 5e-05
        h.cao0_ca_ion = 2
#         h.cli0_cl_ion = 0
#         h.clo0_cl_ion = 0


# Read Input Files
df_tree1 = read_dat('tree1-nom.dat')
df_tree0 = read_dat('tree0-nom.dat')
df_dis_tree1 = pd.read_csv('sth/sth-data/Tree_1_length.csv')['Tree 1']
df_dis_tree0 = pd.read_csv('sth/sth-data/Tree_0_length.csv')['Tree 0']


# Building Cell
class CreateSth():
    def __init__(self, params=None):
        self.scaler = f_min
        self.rhoa= 7e5
        self.soma = h.Section(name='soma', cell=self)
        self.initseg = h.Section(name='initseg', cell=self)
        self._setup_morphology()
        self.sdi = [self.soma] + self.dend0 + self.dend1 + [self.initseg]
        self.axon = self.node + self.MYSA + self.FLUT + self.STIN
        self.all = [self.soma] + self.dend0 + self.dend1 + [self.initseg] + self.node + self.MYSA + self.FLUT + self.STIN
        self._setup_biophysics()
        
    def _setup_morphology(self):
        self.soma.diam = 18.3112
        self.soma.L = 18.8
        self.initseg.diam = 1.8904976874853334
        self.initseg.L = 21.7413353424173
        self.dend1 = []
        self.dend0 = []
        self.node = []
        self.MYSA = []
        self.FLUT = []
        self.STIN = []
        
        # Building Tree 1
        for i in range(11):
            name_str = 'dend1{}'.format(i)
            self.dend1.append(h.Section(name=name_str, cell=self))
            result = find_dat(i, df_tree1)
            diam = result[1]
            L = result[2]
            nseg = result[3]
            self.dend1[i].diam = diam
            self.dend1[i].L = L
            self.dend1[i].nseg = nseg
        
        for i in range(11):
            if i == 0:
                self.dend1[i].connect(self.soma, 0)
                result = find_dat(i, df_tree1)
                children = result[0]
                self.dend1[1].connect(self.dend1[0])
                self.dend1[2].connect(self.dend1[0])
            else:
                result = find_dat(i, df_tree1)
                children = result[0]
                if children != [-1, -1]:
                    self.dend1[children[0]].connect(self.dend1[i])
                    self.dend1[children[1]].connect(self.dend1[i])
        
        # Building Tree 0
        for i in range(23):
            name_str = 'dend0{}'.format(i)
            self.dend0.append(h.Section(name=name_str, cell=self))
            result = find_dat(i, df_tree0)
            diam = result[1]
            L = result[2]
            nseg = result[3]
            self.dend0[i].diam = diam
            self.dend0[i].L = L
            self.dend0[i].nseg = nseg
        for i in range(23):
            if i == 0:
                self.dend0[0].connect(self.soma, 1)
                result = find_dat(i, df_tree0)
                children = result[0]
                self.dend0[children[0]].connect(self.dend0[i])
                self.dend0[children[1]].connect(self.dend0[i])
            else:
                result = find_dat(i, df_tree0)
                children = result[0]
                if children != [-1, -1]:
                    self.dend0[children[0]].connect(self.dend0[i])
                    self.dend0[children[1]].connect(self.dend0[i])
                    
        # Building Axon
        axonnodes=10
        paranodes1=(axonnodes-1)*2
        paranodes2=(axonnodes-1)*2
        axoninter=(axonnodes-1)*3
        
        for i in range(axonnodes):
            name_str = 'node{}'.format(i)
            self.node.append(h.Section(name=name_str, cell=self))
            self.node[i].diam = 1.4
            self.node[i].L = 1
            self.node[i].nseg = 1
        for i in range(paranodes1):
            if i < 6:
                name_str = 'MYSA'
                self.MYSA.append(h.Section(name=name_str, cell=self))
                self.MYSA[i].diam = 1.4
                self.MYSA[i].L = 1.5
                self.MYSA[i].nseg = 1
            else:
                name_str = 'MYSA'
                self.MYSA.append(h.Section(name=name_str, cell=self))
                self.MYSA[i].diam = 1.4
                self.MYSA[i].L = 3
                self.MYSA[i].nseg = 1
        for i in range(paranodes2):
            if i < 6:
                name_str = 'FLUT'
                self.FLUT.append(h.Section(name=name_str, cell=self))
                self.FLUT[i].diam = 1.6
                self.FLUT[i].L = 5
                self.FLUT[i].nseg = 1
            else:
                name_str = 'FLUT'
                self.FLUT.append(h.Section(name=name_str, cell=self))
                self.FLUT[i].diam = 1.6
                self.FLUT[i].L = 10
                self.FLUT[i].nseg = 1
        for i in range(axoninter):
            if i < 9:
                name_str = 'STIN'
                self.STIN.append(h.Section(name=name_str, cell=self))
                self.STIN[i].diam = 1.6
                self.STIN[i].L = 29
                self.STIN[i].nseg = 1
            else:
                name_str = 'STIN'
                self.STIN.append(h.Section(name=name_str, cell=self))
                self.STIN[i].diam = 1.6
                self.STIN[i].L = 58
                self.STIN[i].nseg = 1
        
        # Connecting axon and soma
        self.initseg.connect(self.soma, 1)
        self.node[0].connect(self.initseg, 1)
        self.MYSA[0].connect(self.node[0], 1)
        self.FLUT[0].connect(self.MYSA[0], 1)
        self.STIN[0].connect(self.FLUT[0], 1)
        self.STIN[1].connect(self.STIN[0], 1)
        self.STIN[2].connect(self.STIN[1], 1)
        self.FLUT[1].connect(self.STIN[2], 1)
        self.MYSA[1].connect(self.FLUT[1], 1)
        self.node[1].connect(self.MYSA[1], 1)
        for i in [1,2,3,4,5,6,7,8]:
            self.MYSA[2*i].connect(self.node[i], 1)
            self.FLUT[2*i].connect(self.MYSA[2*i], 1)
            self.STIN[3*i].connect(self.FLUT[2*i], 1)
            self.STIN[3*i+1].connect(self.STIN[3*i], 1)
            self.STIN[3*i+2].connect(self.STIN[3*i+1], 1)
            self.FLUT[2*i+1].connect(self.STIN[3*i+2], 1)
            self.MYSA[2*i+1].connect(self.FLUT[2*i+1], 1)
            self.node[i+1].connect(self.MYSA[2*i+1], 1)

    def _setup_biophysics(self):
        for sec in self.sdi:
            sec.Ra = self.scaler[18]    # Axial resistance in Ohm * cm
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2
            insert_channels(sec)
        for sec in self.axon:
            sec.Ra = self.rhoa / 10000
            sec.cm = 2
        # print("Testing: {}".format(self.scaler))
        for seg in self.soma:
            seg.NaL.gna = self.scaler[7]
            seg.Na.gna = self.scaler[8]
            seg.HVA.gcaL = self.scaler[0]
            seg.HVA.gcaN = self.scaler[1]
            seg.CaT.gcaT = self.scaler[2]
            seg.Ih.gk = self.scaler[3]
            seg.KDR.gk = self.scaler[4]
            seg.Kv31.gk = self.scaler[5]
            seg.sKCa.gk = self.scaler[6]
            seg.STh.gpas = self.scaler[17]
        for seg in self.initseg:
            seg.NaL.gna = self.scaler[7]
            seg.Na.gna = self.scaler[8] * self.scaler[19]
            seg.HVA.gcaL = self.scaler[0]
            seg.HVA.gcaN = self.scaler[1]
            seg.CaT.gcaT = self.scaler[2]
            seg.Ih.gk = self.scaler[3]
            seg.KDR.gk = self.scaler[4]
            seg.Kv31.gk = self.scaler[5]
            seg.sKCa.gk = self.scaler[6]
            seg.STh.gpas = self.scaler[17]
        proximal = 359/2
        dend_i = 0
        for sec in self.dend0:
            for seg in sec:
                if df_dis_tree0[dend_i] < proximal:
                    seg.NaL.gna = self.scaler[15] * self.scaler[7]
                    seg.Na.gna = self.scaler[16] * self.scaler[8]
                    seg.HVA.gcaL = self.scaler[9] * self.scaler[0]
                    seg.HVA.gcaN = self.scaler[10]
                    seg.CaT.gcaT = self.scaler[11]
                    seg.Ih.gk = self.scaler[3]
                    seg.KDR.gk = self.scaler[12] * self.scaler[4]
                    seg.Kv31.gk = self.scaler[13] * self.scaler[5]
                    seg.sKCa.gk = self.scaler[14] * self.scaler[6]
                    seg.STh.gpas = self.scaler[17]
                else:
                    seg.NaL.gna = 0         # could be uniform
                    seg.Na.gna = 0          # could be uniform
                    seg.HVA.gcaL = 0
                    seg.HVA.gcaN = self.scaler[10] # more in dendrites than soma
                    seg.CaT.gcaT = 0     # could be throughout entire dendrite
                    seg.Ih.gk = self.scaler[3]
                    seg.KDR.gk = 0
                    seg.Kv31.gk = 0
                    seg.sKCa.gk = self.scaler[14] * self.scaler[6]
                    seg.STh.gpas = self.scaler[17]
                dend_i += 1
        # print('tree0 total count: {}'.format(dend_i))
        # test_dis = []
        dend_i = 0
        for sec in self.dend1:
            for seg in sec:
                if df_dis_tree1[dend_i] < proximal:
                    seg.NaL.gna = self.scaler[15] * self.scaler[7]
                    seg.Na.gna = self.scaler[16] * self.scaler[8]
                    seg.HVA.gcaL = self.scaler[9] * self.scaler[0]
                    seg.HVA.gcaN = self.scaler[10]
                    seg.CaT.gcaT = self.scaler[11]
                    seg.Ih.gk = self.scaler[3]
                    seg.KDR.gk = self.scaler[12] * self.scaler[4]
                    seg.Kv31.gk = self.scaler[13] * self.scaler[5]
                    seg.sKCa.gk = self.scaler[14] * self.scaler[6]
                    seg.STh.gpas = self.scaler[17]
                else:
                    seg.NaL.gna = 0
                    seg.Na.gna = 0
                    seg.HVA.gcaL = 0
                    seg.HVA.gcaN = self.scaler[10] # more in dendrites than soma
                    seg.CaT.gcaT = 0      # could be throughout entire dendrite
                    seg.Ih.gk = self.scaler[3]
                    seg.KDR.gk = 0
                    seg.Kv31.gk = 0
                    seg.sKCa.gk = self.scaler[14] * self.scaler[6]
                    seg.STh.gpas = self.scaler[17]
                dend_i += 1
        
        for count, sec in enumerate(self.node):
            sec.Ra = self.rhoa/10000
            sec.cm = 2
            sec.insert('extracellular')
            for i in range(1):
                    sec.xraxial[i]=Rpn0
                    sec.xg[i] = 1e10
                    sec.xc[i]=0
            if count == 9:
                sec.insert('pas')
                for seg in sec:
                    seg.pas.g = 0.0001
                    seg.pas.e = -65
            else:
                sec.insert('axnode75')
                for seg in sec:
                    seg.axnode75.gnabar = 2.0       # 3
                    seg.axnode75.gnapbar = 0.05
                    seg.axnode75.gkbar = 0.07
                    seg.axnode75.gl = 0.005
                    seg.axnode75.ek = -85         # -95
                    seg.axnode75.ena = 55         # 45
                    seg.axnode75.el = -60
                    # seg.axnode75.vshift = 15      # 10 # rest potential should be -80+vshift
                    # seg.axnode75.vtraub = -80
        
        for sec in self.MYSA:
            sec.Ra = self.rhoa/10000
            sec.cm = 2
            sec.insert('pas')
            sec.insert('extracellular') # for the myelin
            for seg in sec:
                seg.pas.g = 0.0001
                seg.pas.e = -65
            sec.xraxial[0]=Rpn1
            sec.xg[0] = mygm/(nl*2)
            sec.xc[0]= mycm/(nl*2)

        for sec in self.FLUT:
            sec.Ra = self.rhoa/10000
            sec.cm = 2
            sec.insert('parak75')
            sec.insert('pas')
            sec.insert('extracellular')
            for seg in sec:
                seg.parak75.gkbar = 0.02
                seg.parak75.ek = -85       #-95
                # seg.parak75.vshift = 15    #10
                seg.pas.g = 0.0001
                seg.pas.e = -60
            for i in range(1):
                sec.xraxial[i]=Rpn2
                sec.xg[i] = mygm/(nl*2)
                sec.xc[i]= mycm/(nl*2)
        
        for sec in self.STIN:
            sec.Ra = self.rhoa/10000
            sec.cm = 2
            sec.insert('pas')
            sec.insert('extracellular')
            for seg in sec:
                seg.pas.g = 0.0001
                seg.pas.e = -65
            for i in range(1):
                sec.xraxial[i]=Rpx
                sec.xg[i] = mygm/(nl*2)
                sec.xc[i]= mycm/(nl*2)
            
        for sec in self.sdi:
            h.ion_style("na_ion",1,2,1,0,1, sec=sec)
            h.ion_style("k_ion",1,2,1,0,1, sec=sec)
            h.ion_style("ca_ion",3,2,1,1,1, sec=sec)

    def __repr__(self):
        return 'sth'

