#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github
@mail: ylxx@live.com
Created on Sun Mar 10 16:00:49 2019
"""
from boxx import *
from boxx.ylth import *
from boxx.ylth import tht, map2, npa, th, np
from pylab import plt

nn.Softmax
th.Tensor.softmax

plott = lambda *l, **kv:plt.plot(*map2(lambda t: npa(t) if isinstance(t, th.Tensor) else t, l),**kv)

npoint = 500

xrange = -4, 3
#yrange = -2, 3
yrange = xrange
xs = tht-np.linspace(*xrange, npoint)
ys = tht-np.linspace(*yrange, npoint)

plt.xlim(*xrange) 
plt.ylim(*yrange) 

xs.requires_grad = True

t = 1

plott(xs, xs==1000000.11,'k')
plott(xs==1000000.11, ys, 'k')

for it,t in enumerate([1,.5,3][:2]):
    color = 'bgry'[it]
    
    prob = 1/(1+th.exp(-xs*t))
    loss = -th.log(prob)
    allloss = loss.sum()
    allloss.backward()
    
    grad = xs.grad
    plott(xs, prob, color)
    plott(xs, loss, '--'+color)
    plott(xs, grad, ','+color )
    xs.grad.data.zero_()

plt.grid()
ax = plt.gca()
ax.set_aspect(1)

if __name__ == "__main__":
    pass
    
    
    
