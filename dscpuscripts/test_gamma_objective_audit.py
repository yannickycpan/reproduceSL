"""Fresh gradient checks against the actual GammaTD implementation."""
import copy
import pytest
import torch
from torch import nn
from fd001_gamma.objectives import GammaTD
from agents.fulltd_objectives import TDObjective

class Mean(nn.Module):
    feature_dim=64
    output_dim=1
    def __init__(self):
        super().__init__();self.hidden=nn.Linear(24,64);self.head=nn.Linear(64,1)
    def features(self,x):return torch.tanh(self.hidden(x[:,:,-1]))
    def forward(self,x):return self.head(self.features(x))

def inputs():
    torch.manual_seed(123)
    x=torch.randn(5,24,30,dtype=torch.float64)
    y=torch.randn(5,1,dtype=torch.float64)
    return dict(x=x,y=y,mask=torch.ones_like(y,dtype=torch.bool)),dict(x=.4*x-.1,y=.8*y+.2,mask=torch.ones_like(y,dtype=torch.bool)),torch.arange(5)%2

@pytest.mark.parametrize('variant,n', [('typed',2),('raw26',26),('representation',130)])
def test_initial_mse_and_parameter_partition(variant,n):
    b,p,t=inputs();obj=GammaTD(Mean().double(),variant).double()
    loss,count=obj.loss_sum(b,p,t)
    torch.testing.assert_close(loss/count,.5*(b['y']-obj.model(b['x'])).square().mean())
    assert sum(x.numel() for x in obj.operator_parameters())==n
    groups=obj.parameter_groups(.001,.0003)
    a,c=({id(x) for x in g['params']} for g in groups)
    assert a.isdisjoint(c) and a|c=={id(x) for x in obj.parameters()}

@pytest.mark.parametrize('variant',['typed','raw26','representation'])
def test_all_operator_coordinates_and_predictor_finite_differences(variant):
    b,p,t=inputs();obj=GammaTD(Mean().double(),variant).double()
    with torch.no_grad():
        for q in obj.operator_parameters():q.copy_(torch.linspace(-.02,.03,q.numel()).reshape(q.shape))
    loss,count=obj.loss_sum(b,p,t);(loss/count).backward()
    for name,param in obj.named_parameters():
        indexes=range(param.numel()) if not name.startswith('model.') else [0,param.numel()-1]
        analytic=param.grad.flatten().clone()
        for i in indexes:
            with torch.no_grad():
                x=param.flatten();old=x[i].item();x[i]=old+1e-6
            plus=obj.loss_sum(b,p,t)[0].item()/count
            with torch.no_grad():x[i]=old-1e-6
            minus=obj.loss_sum(b,p,t)[0].item()/count
            with torch.no_grad():x[i]=old
            assert analytic[i].item()==pytest.approx((plus-minus)/2e-6,rel=1e-5,abs=2e-9),(name,i)

def test_typed_is_original_td_bitwise_over_updates():
    b,p,t=inputs();new=GammaTD(Mean().double(),'typed').double();old=TDObjective(copy.deepcopy(new.model),2).double()
    opts=[torch.optim.Adam(o.parameter_groups(.001,.001)) for o in (new,old)]
    for _ in range(5):
        for o,opt in zip((new,old),opts):
            opt.zero_grad();loss,count=o.loss_sum(b,p,t);(loss/count).backward();opt.step()
        torch.testing.assert_close(new.operator,old.theta,rtol=0,atol=0)
        for x,y in zip(new.model.parameters(),old.model.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)

def test_raw26_time_index_sign_and_representation_source_path():
    b,p,t=inputs();raw=GammaTD(Mean().double(),'raw26').double()
    phi=raw.raw_features(b['x'],p['x'],t)
    assert phi.shape==(5,26)
    torch.testing.assert_close(phi[:,2:],b['x'][:,:,-1]-p['x'][:,:,-1])
    obj=GammaTD(Mean().double(),'representation').double()
    with torch.no_grad():obj.gamma_head.weight.fill_(.01)
    h=obj.model.features(b['x']);gamma=obj.gamma_values(b['x'],p['x'],t,hidden=h)
    g,=torch.autograd.grad(gamma.sum(),obj.model.hidden.weight)
    assert torch.linalg.norm(g)>0
