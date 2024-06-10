import pytest
import torch
import torch.nn as nn
from nemo.collections.llm import fn


class CustomMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x):
        return x + self.linear2(self.linear1(x))


class SharedMLP(nn.Module):
    def __init__(self, shared: nn.Module):
        super().__init__()
        self.linear1 = shared
        self.linear2 = shared

    def forward(self, x):
        return x + self.linear2(self.linear1(x))


def add_relu(x):
    if isinstance(x, nn.Linear):
        return nn.Sequential(x, nn.ReLU())
    return x


def add_relu_named(x, name=None, to_replace="linear1"):
    if name == to_replace and isinstance(x, nn.Linear):
        return nn.Sequential(x, nn.ReLU())
    return x


def add_relu_first(x, i=None):
    if i == 0 and isinstance(x, nn.Linear):
        return nn.Sequential(x, nn.ReLU())
    return x


class TestWalkModule:
    def test_map_identity(self):
        # Test mapping an identity function
        module = nn.Linear(10, 10)
        identity = lambda x: x
        assert fn.map(module, identity) is module

    def test_map_transform(self):
        # Test mapping a transform function
        module = nn.Linear(10, 10)
        transformed_module = fn.map(module, add_relu)
        assert isinstance(transformed_module[0], nn.Linear)
        assert isinstance(transformed_module[1], nn.ReLU)

    def test_walk_custom_module(self):
        mlp = CustomMLP()
        with_relu = fn.walk(mlp, add_relu)
        assert isinstance(with_relu.linear1, nn.Sequential)
        assert isinstance(with_relu.linear2, nn.Sequential)

        for walk_fn in [add_relu_named, add_relu_first]:
            with_relu_first = fn.walk(CustomMLP(), walk_fn)
            assert isinstance(with_relu_first.linear1, nn.Sequential)
            assert isinstance(with_relu_first.linear2, nn.Linear)

    def test_walk_shared_module(self):
        def double_linear(module: nn.Module):
            if isinstance(module, nn.Linear):
                module.weight.data *= 2
                module.bias.data *= 2
            return module

        shared_linear = nn.Linear(10, 10)
        mlp = SharedMLP(shared_linear)

        # Get initial weight and bias values
        initial_weight = shared_linear.weight.data.clone()
        initial_bias = shared_linear.bias.data.clone()

        # Apply the doubling function using walk
        transformed_mlp = fn.walk(mlp, double_linear)

        # Check that the shared linear module was only transformed once
        assert torch.allclose(transformed_mlp.linear1.weight.data, initial_weight * 2)
        assert torch.allclose(transformed_mlp.linear1.bias.data, initial_bias * 2)
        assert torch.allclose(transformed_mlp.linear2.weight.data, initial_weight * 2)
        assert torch.allclose(transformed_mlp.linear2.bias.data, initial_bias * 2)
        assert transformed_mlp.linear1 is transformed_mlp.linear2

    def test_leaf_only(self):
        def is_linear(module: nn.Module):
            assert isinstance(module, nn.Linear)

            return module

        fn.walk(CustomMLP(), is_linear, leaf_only=True)


class TestWalkListModule:
    @pytest.mark.parametrize("module_container", [nn.ModuleList, nn.Sequential])
    def test_walk_module_container(self, module_container):
        modules = [nn.Linear(10, 10), nn.Linear(10, 10)]
        module = module_container(modules) if module_container is nn.ModuleList else nn.Sequential(*modules)

        def walk_fn(module):
            if isinstance(module, nn.Linear):
                module.weight.data.fill_(1.0)
            return module

        walked_module = fn.walk(module, walk_fn)

        assert isinstance(walked_module, module_container)
        assert len(walked_module) == 2
        assert torch.allclose(walked_module[0].weight, torch.ones_like(walked_module[0].weight))
        assert torch.allclose(walked_module[1].weight, torch.ones_like(walked_module[1].weight))

    @pytest.mark.parametrize("module_container", [nn.ModuleList, nn.Sequential])
    def test_walk_module_container_with_kwargs(self, module_container):
        modules = [nn.Linear(10, 10), nn.Linear(10, 10)]
        module = module_container(modules) if module_container is nn.ModuleList else nn.Sequential(*modules)

        def walk_fn(module, value):
            if isinstance(module, nn.Linear):
                module.weight.data.fill_(value)
            return module

        walked_module = fn.walk(module, walk_fn, value=2.0)

        assert isinstance(walked_module, module_container)
        assert len(walked_module) == 2
        assert torch.allclose(walked_module[0].weight, 2.0 * torch.ones_like(walked_module[0].weight))
        assert torch.allclose(walked_module[1].weight, 2.0 * torch.ones_like(walked_module[1].weight))

    @pytest.mark.parametrize("module_container", [nn.ModuleList, nn.Sequential])
    def test_walk_module_container_with_recursion(self, module_container):
        modules = [
            nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10)),
            nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10)),
        ]
        module = module_container(modules) if module_container is nn.ModuleList else nn.Sequential(*modules)

        def walk_fn(module):
            if isinstance(module, nn.Linear):
                module.weight.data.fill_(1.0)
            return module

        walked_module = fn.walk(module, walk_fn)

        assert isinstance(walked_module, module_container)
        assert len(walked_module) == 2
        for seq in walked_module:
            assert isinstance(seq, nn.Sequential)
            assert len(seq) == 2
            assert torch.allclose(seq[0].weight, torch.ones_like(seq[0].weight))
            assert torch.allclose(seq[1].weight, torch.ones_like(seq[1].weight))


class TestWalkDictModule:
    def test_walk_module_dict_identity(self):
        """
        Test walking through an nn.ModuleDict without applying any transformations,
        essentially testing the identity operation.
        """
        # Setup
        modules = nn.ModuleDict({"linear": nn.Linear(10, 10), "conv": nn.Conv2d(1, 20, 5)})
        identity = lambda x: x

        # Exercise
        walked_modules = fn.walk(modules, identity)

        # Verify
        assert isinstance(walked_modules, nn.ModuleDict)
        assert "linear" in walked_modules and isinstance(walked_modules["linear"], nn.Linear)
        assert "conv" in walked_modules and isinstance(walked_modules["conv"], nn.Conv2d)

    def test_walk_module_dict_transform(self):
        """
        Test walking through an nn.ModuleDict and applying a transformation to each module.
        In this case, we'll add a ReLU activation after each module.
        """
        modules = nn.ModuleDict({"linear": nn.Linear(10, 10), "conv": nn.Conv2d(1, 20, 5)})

        def add_relu(module: nn.Module, name=None):
            if name in ["linear", "conv"]:
                return nn.Sequential(module, nn.ReLU())

            return module

        walked_modules = fn.walk(modules, add_relu)
        assert isinstance(walked_modules, nn.ModuleDict)
        for module in walked_modules.values():
            assert isinstance(module, nn.Sequential)
            assert isinstance(module[1], nn.ReLU)
