import tensorflow as tf


def distribute(model, devices):
    class DistributedModel(model.__class__):
        def inference(self, optimizee, input_state, n_bptt_steps):
            inference = {}
            scope = tf.get_variable_scope()
            for dev in self.devices:
                with tf.device(dev):
                    inference[dev] = super(DistributedModel, self).inference(optimizee, input_state, n_bptt_steps)
                    if not scope.reuse:
                        scope.reuse_variables()

            return inference


        def loss(self, inference, **kwargs):
            losses = {
                dev: super(DistributedModel, self).loss(inf, **kwargs)
                for dev, inf in inference.items()
            }
            return losses


        def grads(self, optimizer, losses):
            tower_grads = []
            for dev in self.devices:
                with tf.device(dev):
                    grads = super(DistributedModel, self).grads(optimizer, losses[dev])
                    tower_grads.append(grads)

            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                grad = tf.stack([g for g, _ in grad_and_vars], axis=0)
                grad = tf.reduce_mean(grad, axis=0)

                v = grad_and_vars[0][1]
                average_grads.append((grad, v))

            return average_grads

    model.__class__ = DistributedModel
    model.devices = devices
    return model
