# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 03:27:13 2019

@author: yl

1. getMaskCache and getTempleCache
2. pce: pointwise coress entropy in in cyc
"""
from boxx import *
if not cloud:
    from boxx.ylth import *
import torch as th
import torch
from boxx import Vector, getWeightCore, resize, np, sliceLimit, timegap, timeit, dicto, map2, strnum, Markdown, cf
import torch
# papre output and gt

eps = 1e-6
def pasteImg(bg, fg, startPoint=None):
    '''
    paste @fg to @bg at @startPoint
    
    Parameters
    ----------
    bg : ndarray or torch.tensor
        background
    fg : ndarray or torch.tensor
        foreground
    startPoint : boxx.Vector 'hw' format, default Vector([0, 0])
        start point
        
    Notice
    ----------
    1. startePoint is Vector([h, w])
    2. bg will be changed by paste
    '''
    if startPoint is None:
        startPoint = Vector([0, 0])
    fghw = Vector(fg.shape[-2:])
    newPatch = sliceLimit(bg)[...,startPoint.h :max(0, startPoint.h+fghw.h), startPoint.w :max(0,startPoint.w+fghw.w)]
    newhw = Vector(newPatch.shape[-2:])
    
    bgIndex = sliceLimit(bg, True)[...,startPoint.h :max(0, startPoint.h)+newhw.h, startPoint.w :max(0, startPoint.w)+newhw.w]
    fgIndex = sliceLimit(fg, True)[...,-startPoint.h :max(0, -startPoint.h) + newhw.h, -startPoint.w :max(0, -startPoint.w) + newhw.w]
    
#    g()
    bg[bgIndex] = fg[fgIndex]
    return bg
import torch.nn as nn

def gradLoss(feat, pgt, c_max, c_step, ):
    pass

#cf.debugPoinMax = 2
class PointMax(nn.Module):
    def __init__(self, w=1, suppressionBg=False):
        super(PointMax, self).__init__()
        self.w = w
        self.suppressionBg = suppressionBg
    def forward(self, feats, xyens):
        shape = feats.shape
        device = feats.device
        dim1, dim2 = np.mgrid[:shape[-4],:shape[-3]]
        dim1, dim2 = torch.tensor(dim1.ravel(), device=device), torch.tensor(dim2.ravel(), device=device)
        xyens = xyens.view(-1, 3)
        typee = feats.type()
        x, y, existMask = xyens[...,0].type(th.long).to(device), xyens[...,1].type(th.long).to(device), xyens[...,-1].type(typee).to(device)
#        tree([dim1.ravel(), dim2.ravel(), y, x], deep=1)
        validXMask = (x>=0)&(x<shape[-1])
        validYMask = (y>=0)&(y<shape[-2])
        
#        x = torch.clamp(x, 0, shape[-1])
#        y = torch.clamp(y, 0, shape[-2])
        x[~validXMask] = 0
        y[~validYMask] = 0
        
        indMask = ((existMask>0) & validXMask & validYMask).type(typee)
        
        point_feats = feats[dim1, dim2, y, x]
        siged = th.sigmoid(point_feats)
        loss = -th.log(siged + eps)
        loss = (loss*indMask).sum()/(indMask.sum()+eps)
        loss *= self.w
        
        if self.suppressionBg or cf.get('debugPoinMax'):
            backloss = self._suppressionBg(feats, loss)
        
            s = f"pointmax: {loss} + "
            loss += backloss
            s += f"backloss: {backloss}"
            s = f"loss: {loss} = " + s
            if timegap(cf.debugPoinMax, 'debugPoinMax'):
                pred-s
#        g()
#        1/0
        return loss
    def _suppressionBg(self, feats):
        backsiged = th.sigmoid(feats.mean(-1).mean(-1))
        backloss = -th.log(1-backsiged + eps)
        backloss = backloss.mean()*self.w
        return backloss
    
    
def getMaskOfPgt(argkv):
    pgt, cycle_tmpl, shape, out_cyc_r = argkv.get('pgt'), argkv.get('cycle_tmpl'), argkv.get('shape'), argkv.get('out_cyc_r')
    if not isinstance(pgt, Vector):
        pgt = Vector(pgt)
    bg = np.zeros(shape[-2:], dtype=np.int32) 
    if out_cyc_r:
        bg -= 1
    mask = pasteImg(bg, cycle_tmpl, pgt-cycle_tmpl.shape[-1]//2)
    return (mask==0), (mask==1), 

def layerNormaFun(feat):
    mean =( feat.mean(-1, True).mean(-1, True))
#    return (feat-mean)/((feat-mean)**2).mean(-1, True).mean(-1, True)**.5
    return (feat-mean)/((((feat-mean)**2).mean(-1, True).mean(-1, True)+eps)**.5 + eps)

class SpatialSoftmax(nn.Module):
    def __init__(self, cyc_r=7, log_freq=60*5, poolings=['avg'], number_worker=8, out_cyc_r=None, layerNorma=True, probMargin=None, temper=1):
        super(SpatialSoftmax, self).__init__()
        self.out_cyc_r = out_cyc_r
        self.poolings = poolings
        self.log_freq = log_freq
        self.cyc_r = cyc_r
        self.layerNorma = layerNorma
        self.probMargin = probMargin
        self.temper = temper
        raw_cycle_tmpl = getWeightCore(300, seta=.25)
        raw_cycle_tmpl = raw_cycle_tmpl > raw_cycle_tmpl[150,0]
        cycle_tmpl_np = resize(raw_cycle_tmpl, np.int8((cyc_r*2, cyc_r*2, )))
        cycle_tmpl = np.int32(cycle_tmpl_np > .5)
        if out_cyc_r is not None:
            out_cycle_tmpl_np = resize(raw_cycle_tmpl, (out_cyc_r*2, out_cyc_r*2, ))
            out_cycle_tmpl = np.int32(out_cycle_tmpl_np >.5) - 1
            inCycIndx = np.zeros(out_cycle_tmpl.shape, np.bool)
            pasteImg(inCycIndx, cycle_tmpl , Vector([out_cyc_r-cyc_r]*2))
            out_cycle_tmpl[inCycIndx] = 1
            cycle_tmpl = out_cycle_tmpl
#        self.cycle_tmpl =  th.from_numpy(cycle_tmpl)  
        self.cycle_tmpl = (cycle_tmpl)  
#    def getMasks(self, pgts):
#        masks = []
#        for pgt in pgts:
#            mask = self.getMask(pgt)
#            masks.append(mask[None])
#        return th.cat(masks)>0
#    def forward_single_threading(self, feats, pgtns):
#        def softmaxAB(a, b,):
#            expa = th.exp(a)
#            expb = th.exp(b)
#            return expa/(expa+expb+eps)
#        
#        self.shape = feats.shape
#        losses = []
#        for batchind, (feat, pgts) in enumerate(zip(feats, pgtns)):
#            for featNode, pgt in zip(feat, pgts):
#                if not isinstance(pgt, Vector):
#                    pgt = Vector(pgt.cpu())
#                inMask, outMask = self.getMask(pgt)
#                inCycs = featNode[inMask]
#                outCycs = featNode[outMask]
#                for pool in self.poolings:
#                    pool = {'avg':th.mean, 'max':th.max}[pool]
#                    inCyc, outCyc = pool(inCycs), pool(outCycs)
#                    prob = softmaxAB(inCyc, outCyc)
#                    loss = -th.log(prob + eps)
#                    losses += [loss]
#        loss = sum(losses)/len(losses)
#        return loss
    def forward(self, feats, xyens):
        logName = 'cyc_r: %s, out_cyc_r: %s'%(self.cyc_r, self.out_cyc_r or 'N')
        logTag = timegap(self.log_freq, logName)
        if logTag:
            logDic = dicto()
        if self.layerNorma:
            feats = layerNormaFun(feats)
        tensorType = feats.type()
        shape = self.shape = feats.shape
        feats = feats.view(-1, *shape[-2:])
        xyens = xyens.view(-1, 3)
        pgts = xyens[...,[1,0]].cpu().numpy()
        existMask = xyens[...,-1].type(tensorType)
        
#        with timeit(logName):
        masks = map2(getMaskOfPgt, [dict(pgt=pgt, cycle_tmpl=self.cycle_tmpl, shape=shape, out_cyc_r=self.out_cyc_r) for pgt in pgts])
#            masks = mapmp(self.getMask, pgts, pool=4)
        masks = np.array(masks)
        masks = th.from_numpy(np.uint8(masks)).type(tensorType).cuda()
        
        loss = 0
        #(lambda a,b,t=1:e**(t*a)/(e**(t*a)+e**(t*b)))(2,1,5)
        
        
        def softmaxFgBg(fg, bg, t=1):
            fge = th.exp(fg*t)
            bge = th.exp(bg*t)
            prob = fge/(fge+bge+eps)
            return prob
        def CE(fg, bg):
            prob = softmaxFgBg(fg, bg)
            avgLosses = -th.log(prob+eps)
            return avgLosses
        
        if 'avg' in self.poolings:
            bgAvgPool = (feats * masks[...,0,:,:]).sum(-1).sum(-1)/(masks[...,0,:,:].sum(-1).sum(-1)+eps)
            fgAvgPool = (feats * masks[...,1,:,:]).sum(-1).sum(-1)/(masks[...,1,:,:].sum(-1).sum(-1)+eps)
            avgProbs = softmaxFgBg(fgAvgPool, bgAvgPool, self.temper)
            avgLosses = -th.log(avgProbs+eps)
            
            indexMask = existMask*(avgProbs<self.probMargin).type(tensorType) if self.probMargin else existMask
            
            avgLoss = (avgLosses*indexMask).sum()/(indexMask.sum()+eps)
            loss += avgLoss
            
            if logTag:
                logDic.avgLoss = float(avgLoss)
                logDic.avgProb = float(avgProbs.mean())
                
        if 'max' in self.poolings:
            bgMaxPool = (feats * masks[...,0,:,:]).max(-1)[0].max(-1)[0]
            fgMaxPool = (feats * masks[...,1,:,:]).max(-1)[0].max(-1)[0]
            maxProbs = softmaxFgBg(fgMaxPool, bgMaxPool, self.temper)
            maxLosses = -th.log(maxProbs+eps)
            
            indexMask = existMask*(maxProbs<self.probMargin).type(tensorType) if self.probMargin else existMask
            
            maxLoss = (maxLosses*indexMask).sum()/(indexMask.sum()+eps)
            loss += maxLoss
            
            if logTag:
                logDic.maxLoss = float(maxLoss)
                logDic.maxProb = float(maxProbs.mean())
        if logTag:
            print("%s | %s"%(logName, ', '.join(map(lambda kv: "%s: %.3f"%kv, logDic.items()))))
#            print(Markdown([{k:strnum(v) for k,v in logDic.items()}]))
#        g()
        return loss
class MultiScaleSpatialSoftmax(nn.Module):
    def __init__(self, cyc_rs=[8, 4, 2, 1], weights=None, log_freq=60*5, poolings=['avg'], out_cyc=True, pointMaxW=1, layerNorma=True, probMargin=None, temper=1):
        self.poolings = poolings
        self.layerNorma = layerNorma
        super(MultiScaleSpatialSoftmax, self).__init__()
        self.cyc_rs = cyc_rs
        self.log_freq = log_freq
        self.spatialSoftmaxs = []
        self.temper = temper
        
        out_cyc_r = None
        for r in cyc_rs:
            ssm = SpatialSoftmax(r, log_freq=log_freq, poolings=poolings, out_cyc_r=out_cyc_r, layerNorma=False, probMargin=probMargin, temper=temper)
            self.spatialSoftmaxs.append(ssm)
            if out_cyc:
                out_cyc_r = r
        if weights is None:
            weights = [1] * len(cyc_rs)
        self.weights = weights
        self.pointMaxW = pointMaxW
        if pointMaxW:
            self.pointMax = PointMax()
        
    def forward(self, feats, xyens):
        
        if self.layerNorma:
            feats = layerNormaFun(feats)
        losses = [spatialSoftmax(feats, xyens)*w for w, spatialSoftmax in zip(self.weights, self.spatialSoftmaxs)]
        if self.pointMaxW:
            losses += [self.pointMaxW*self.pointMax(feats, xyens)]
        if timegap(self.log_freq, 'losses'):
            print(Markdown([dict(zip(self.cyc_rs + ['point'], [strnum(float(loss.cpu())) for loss in losses]))]))
        return sum(losses)/len(losses)

if 0:
    avg = MultiScaleSpatialSoftmax(log_freq=60, out_cyc=True, poolings=['avg',])
    
    avg_max = MultiScaleSpatialSoftmax(log_freq=60, out_cyc=True, poolings=['avg', 'max'])
    
    avg_pmw5 = MultiScaleSpatialSoftmax(log_freq=60, out_cyc=True, poolings=['avg',], pointMaxW=5)
    
    avg_pmw0 = MultiScaleSpatialSoftmax(log_freq=60, out_cyc=True, poolings=['avg',], pointMaxW=0)
    
    avg_pmw0 = MultiScaleSpatialSoftmax(log_freq=60, out_cyc=False, poolings=['avg',])


    
if __name__ == "__main__":
    hw = Vector((64, 64))
    channl = 1
    batch = 1
    multi_batch = False
    multi_batch = True
    if multi_batch:
        gpun = 1
        batchPerGpu = 3
        channl = 2
        if cloud:
            batchPerGpu = 32
            channl = 17
            gpun = th.cuda.device_count()
        batch = batchPerGpu * gpun
        
    cyc_r = intround((25)/np.array([353, 257]).mean()*hw.mean())
    pgt = hw//2
    pshift = (hw*.13).intround()
    pshift = Vector([0,0])
    pre = pgt + pshift 
    coreShape = Vector((cyc_r*2, cyc_r*2, ))
    
    
    feat = np.zeros([batch, channl]+ list(hw))
    raw_cycle_tmpl = getWeightCore(300, seta=.25)
#    raw_cycle_tmpl = raw_cycle_tmpl > raw_cycle_tmpl[150,0]
    core = resize(raw_cycle_tmpl, coreShape)
    
    msegt = feat.copy()
    msegt[..., (pgt-coreShape//2).x:(pgt+coreShape//2).x,(pgt-coreShape//2).y:(pgt+coreShape//2).y,] = core
    feat[..., (pre-coreShape//2).x:(pre+coreShape//2).x,(pre-coreShape//2).y:(pre+coreShape//2).y,] = core
    
    
    msegt = torch.tensor(msegt)
    feat = torch.tensor(feat)
    
#    show(core, feat)
    
    
    cert = SpatialSoftmax(log_freq=20, )
    cert = MultiScaleSpatialSoftmax(log_freq=1, out_cyc=True, poolings=['avg', 'max'], pointMaxW=1, layerNorma=True, probMargin=.5)
#    cert = MultiScaleSpatialSoftmax(log_freq=60, out_cyc=True, poolings=['avg',], pointMaxW=0)
#    cert = PointMax() # 8gpu : 0.1332138
    
    if cloud:
        cert = nn.DataParallel(cert)
        # 4 gpus before 1.8 after 0.9
        
        # MultiScaleSpatialSoftmax old
        # 4 gpus: real is 7s, test is
        # 8 gpus: real is 12s, test is 7.6s
        
        
        # MultiScaleSpatialSoftmax + matrix
        # 8 gpus: real is s, test is 2.77 s
        
        # MultiScaleSpatialSoftmax + mapmp
        # 8 gpus: real is s, test is  s
        
        
        # SpatialSoftmax old
        # 8 gpus: real is s, test is 1.8s
        # SpatialSoftmax + mapmp
        # 8 gpus: real is s, test is 0.73 s
    xye = (*pgt[::-1], 1)
    pgtns = th.from_numpy(np.array([[xye]*channl]*batch, np.int64))
    
#    feat *= 1
    
    d = loadData('/home/dl/Downloads/tmp_file_joints_output')
    n = 2
    n = None
    feat, pgtns = tht-d['output'][:n], tht-d['joints'][:n]
    
    pgtns[0][0][:2] = tht-[-2, 99]
    
    feat = feat.cuda()
    feat.requires_grad = True
    for i in range( 1):
#        with timeit('loss'):
        loss = cert(feat, pgtns)
#        loss = feat.mean()
        pred-(loss)
        loss.backward()
    if not multi_batch:
        with timeit('back'):
            loss.backward()
        grad = feat.grad
        tree-(feat.requires_grad, feat.grad)
        assert grad is not None
        loga(grad)
        show(msegt, feat, msegt-feat, grad)
        plot3dSurface(npa-grad[0][0])
        print(loss)
