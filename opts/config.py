class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(self._default)
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self._default})


    @classmethod
    def from_namespace(cls, namespace):
        return cls(**vars(namespace))


class BuildConfig(Config):
    _default = {
        'n_bptt_steps': 20,
        'lambd_l1': 0.,
        'inference_only': False,
        'normalize_lstm_grads': False,
        'grad_clip': 1.
    }


class TestConfig(Config):
    _default = {
        'eid': 0,
        'n_steps': 1000,
        'n_batches': 100,
        'batch_size': 100,
        'restore': True,
        'opt_name': None
    }


class TrainConfig(Config):
    _default = {
        'n_steps': 100,
        'n_epochs': 100,
        'n_batches': 100,
        'batch_size': 100,
        'eid': 0,
        'save_every': 10,
        'test_every': 10,
        'masked_train': 'none',
        'masked_train_p': 0.2,
        'train_lr': 1e-4,
        'momentum': 0.9,
        'test': True
    }


    def to_test_config(self):
        return TestConfig(n_batches=self.n_batches, batch_size=self.batch_size, n_steps=self.n_steps, restore=False)
