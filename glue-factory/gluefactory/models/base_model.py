"""
Base class for trainable models.
"""

from abc import ABCMeta, abstractmethod
from copy import copy

import omegaconf
from omegaconf import OmegaConf
from torch import nn
import torch
import traceback

class MetaModel(ABCMeta):
    def __prepare__(name, bases, **kwds):
        total_conf = OmegaConf.create()
        for base in bases:
            for key in ("base_default_conf", "default_conf"):
                update = getattr(base, key, {})
                if isinstance(update, dict):
                    update = OmegaConf.create(update)
                total_conf = OmegaConf.merge(total_conf, update)
        return dict(base_default_conf=total_conf)


class BaseModel(nn.Module, metaclass=MetaModel):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It recursively updates the default_conf of all parent classes, and
        it is updated by the user-provided configuration passed to __init__.
        Configurations can be nested.

        required_data_keys: list of expected keys in the input data dictionary.

        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unknown configuration entries will raise an error.

        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.

        loss(self, pred, data): method that returns a dictionary of losses,
        computed from model predictions and input data. Each loss is a batch
        of scalars, i.e. a torch.Tensor of shape (B,).
        The total loss to be optimized has the key `'total'`.

        metrics(self, pred, data): method that returns a dictionary of metrics,
        each as a batch of scalars.
    """

    default_conf = {
        "name": None,
        "trainable": True,  # if false: do not optimize this model parameters
        "freeze_batch_normalization": False,  # use test-time statistics
        "timeit": False,  
    }
    required_data_keys = []
    strict_conf = False

    are_weights_initialized = False

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        default_conf = OmegaConf.merge(
            self.base_default_conf, OmegaConf.create(self.default_conf)
        )
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)

        # fixme: backward compatibility
        if "pad" in conf and "pad" not in default_conf:  # backward compat.
            with omegaconf.read_write(conf):
                with omegaconf.open_dict(conf):
                    conf["interpolation"] = {"pad": conf.pop("pad")}

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

        if not conf.trainable:
            for p in self.parameters():
                p.requires_grad = False
        else:
            print(f"93 base_model.py {self.__class__.__name__} is trainable")

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        if self.conf.freeze_batch_normalization:
            self.apply(freeze_bn)

        return self

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""        
        torch.cuda.empty_cache()
        def recursive_key_check(expected, given):
            for key in expected:
                assert key in given, f"Missing key {key} in data"
                if isinstance(expected, dict):
                    recursive_key_check(expected[key], given[key])

        recursive_key_check(self.required_data_keys, data)
        torch.cuda.empty_cache()
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def load_state_dict(self, *args, **kwargs):
        """Load the state dict of the model, and set the model to initialized."""
        torch.cuda.empty_cache()
        ## Adaptations to have RGB, RGB-D, stereo channels in lightglue
        # ret = super().load_state_dict(*args, **kwargs)
        try:
            # Attempt to load the state dictionary as usual
            ret = super().load_state_dict(*args, **kwargs)
        except RuntimeError as e:
            print("\nError loading state_dict:", str(e))

            # Hardcoded solution - 'extractor.conv1a.weight' is first key of suerpoint
            state_dict = args[0] if args else kwargs.get('state_dict', {})
            layer_name = 'extractor.conv1a.weight'

            if layer_name in state_dict:
                # We know the expected input channels need to be expanded from 1 to 3
                first_layer_weights = state_dict[layer_name]
                # grayscale weights has 1 channel
                loaded_channels = 1 
                # Directly access the expected input channels from the model's layer
                if hasattr(self, 'extractor') and hasattr(self.extractor, 'conv1a'):
                    expected_channels = self.extractor.conv1a.weight.size(1)
                    print(f"Expected channels from model structure: {expected_channels}")
                else:
                    print("Could not access expected channels from model structure, defaulting to 3.")
                    expected_channels = 3
                new_weights = first_layer_weights.repeat(1, expected_channels // loaded_channels, 1, 1)
                new_weights /= (expected_channels / loaded_channels)
                state_dict[layer_name] = new_weights

                # Update kwargs or args with the adjusted state_dict for a retry
                if 'state_dict' in kwargs:
                    kwargs['state_dict'] = state_dict
                else:
                    args = (state_dict,) + args[1:]

                # Retry loading the state dictionary
                ret = super().load_state_dict(*args, **kwargs)
                print(f"Adjusted {layer_name} weights from {loaded_channels} to {expected_channels} input channels.")

            self.set_initialized()
            return ret
        else:
            self.set_initialized()
            return ret

    def is_initialized(self):
        """Recursively check if the model is initialized, i.e. weights are loaded"""
        is_initialized = True  # initialize to true and perform recursive and
        for _, w in self.named_children():
            if isinstance(w, BaseModel):
                # if children is BaseModel, we perform recursive check
                is_initialized = is_initialized and w.is_initialized()
            else:
                # else, we check if self is initialized or the children has no params
                n_params = len(list(w.parameters()))
                is_initialized = is_initialized and (
                    n_params == 0 or self.are_weights_initialized
                )
        print(f"164 of base model the is_initalised and end of func {is_initialized}")
        return is_initialized

    def set_initialized(self, to: bool = True):
        """Recursively set the initialization state."""
        self.are_weights_initialized = to
        for _, w in self.named_parameters():
            if isinstance(w, BaseModel):
                w.set_initialized(to)

