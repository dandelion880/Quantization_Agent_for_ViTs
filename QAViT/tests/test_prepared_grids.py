"""Mechanical checks on finite, length-independent frozen grids."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.paths import ROOT
from ptq_agent.execution import grids_partition as grid
from ptq_agent.schemas import PTQError
import torch

class PartitionChecks(unittest.TestCase):
    def test_fixed_groups_across_sequence_lengths_and_transposed_key(self):
        for shape, axis, channel in [((2, 9, 6),1,2),((2, 3, 9, 6),2,3),((2, 3, 6, 9),3,2)]:
            generator = torch.Generator().manual_seed(115)
            x = torch.randn(shape, generator=generator)
            spec, layout = {'bits':4,'axis':channel}, {'axis':axis,'prefix_ends':[1,3]}
            params = grid.fit(x, spec, layout, [1.,.9])
            grid.validate(params,spec,layout,shape)
            actual = grid.apply(x,spec,layout,params)
            longer = torch.cat([x, x],dim=axis)
            output = grid.apply(longer,spec,layout,params)
            self.assertTrue(torch.equal(output.narrow(axis,0,shape[axis]),actual))
            self.assertTrue(torch.equal(grid.apply(x*0,spec,layout,params),x*0))
            for i,(start,stop) in enumerate(grid.spans(longer.shape[axis],layout['prefix_ends'])):
                scaled = output.movedim(axis,1)[:,start:stop] / params['scale'][i] + params['zero'][i]
                self.assertTrue(torch.allclose(scaled, scaled.round(), atol=1e-5))
                self.assertTrue((scaled.round() >= 0).all() and (scaled.round() <= 15).all())
            with self.assertRaises(PTQError):
                grid.validate(params,spec,{'axis':axis,'prefix_ends':[1,9]},shape)
