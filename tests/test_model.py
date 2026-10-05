import unittest
from dataclasses import replace
import torch
from model.reference_model import ModelConfig, KAMoE, KANLinear, ReferenceSoilMoistureModel, reconstruct_sm


def config():
    # Small tensor dimensions for implementation tests, not manuscript settings.
    return ModelConfig(3,2,3,4,3,13,4,6,3,2,2,0.,3,3,-3.,3.,.1,8,8,True,True)


class ModelTests(unittest.TestCase):
    def test_top_one_rejected(self):
        for k in (0,1,7):
            with self.assertRaises(ValueError): KAMoE(3,replace(config(),top_k=k))

    def test_routing_and_router_gradient(self):
        for k in (2,3):
            torch.manual_seed(12)
            module=KAMoE(3,replace(config(),top_k=k))
            x=torch.tensor([[-.4,.2,.9],[.7,-.2,.1]])
            output,aux=module(x,return_aux=True)
            self.assertEqual(tuple(output.shape),(2,3))
            self.assertEqual(tuple(aux['topk_indices'].shape),(2,k))
            torch.testing.assert_close(aux['topk_weights'].sum(1),torch.ones(2))
            output.square().sum().backward()
            self.assertTrue(torch.isfinite(module.gate_out.weight.grad).all())
            self.assertGreater(module.gate_out.weight.grad.abs().sum().item(),1e-8)

    def test_spline_partition_and_shape(self):
        layer=KANLinear(3,2,config())
        x=torch.tensor([[-2.9,0.,2.9]])
        torch.testing.assert_close(layer.b_splines(x).sum(-1),torch.ones_like(x))
        self.assertEqual(tuple(layer(x).shape),(1,2))

    def test_context_and_residual_shapes(self):
        torch.manual_seed(12)
        module=ReferenceSoilMoistureModel(config())
        x=torch.zeros(2,3);index=torch.tensor([0,1])
        out=module(x,index,index,index,index,index+1,index)
        self.assertEqual(tuple(out.shape),(2,1))
        with self.assertRaises(ValueError):module(x)
        prior=torch.tensor([.1,.2]);increment=torch.tensor([[.3],[-.5]])
        torch.testing.assert_close(reconstruct_sm(prior,increment),torch.tensor([[.4],[-.3]]))
        with self.assertRaises(ValueError):reconstruct_sm(prior,torch.zeros(2,2))


if __name__=='__main__':unittest.main()
