import tensorflow as tf
from . import basic_model


class DistributedModel:
    def __init__(self, model, devices):
        self.model = model
        self.devices = devices or ['/cpu:0']


    def __getattr__(self, name):
        return getattr(self.model, name)

    
    def inference(self, optimizee, input_state, n_bptt_steps):
        inference = {}
        scope = tf.get_variable_scope()
        for dev in self.devices:
            with tf.device(dev):
                inference[dev] = self.model.inference(optimizee, input_state, n_bptt_steps)
                if not scope.reuse:
                    scope.reuse_variables()

        return inference


    def loss(self, inference, **kwargs):
        losses = {
            dev: self.model.loss(inf, **kwargs)
            for dev, inf in inference.items()
        }
        return losses


    def grads(self, optimizer, losses):
        tower_grads = []
        for dev in self.devices:
            with tf.device(dev):
                loss = tf.add_n(losses[dev])
                grads = self.model.grads(optimizer, loss)
                tower_grads.append(grads)

        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grad = tf.stack([g for g, _ in grad_and_vars], axis=0)
            grad = tf.reduce_mean(grad, axis=0)

            v = grad_and_vars[0][1]
            average_grads.append((grad, v))

        return average_grads
