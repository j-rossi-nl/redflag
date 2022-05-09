"""
A Learning-Rate schedule for TensorFlow (mimics the cosine with warm-up from transformers PyTorch)

2020. Anonymous authors.
"""

import tensorflow as tf

import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from transformers import AdamWeightDecay


class WarmUpCosineHardRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that follows a Cosine with Hard Restarts schedule, starting with a warmup."""

    def __init__(
            self,
            initial_learning_rate,
            cycle_steps,
            warmup_steps,
            end_learning_rate=1e-6,
            name='CosineWithHardRestarts'):
        """Applies a Cosine with Hard Restarts schedule
        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
          cycle_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive. Length of one cycle in steps
          warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive. Length of the warmup
          end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The minimal end learning rate.
          name: String.  Optional name of the operation. Defaults to
            'CosineWithHardRestarts'.
        Returns:
          A 1-arg callable learning rate schedule that takes the current optimizer
          step and outputs the decayed learning rate, a scalar `Tensor` of the same
          type as `initial_learning_rate`.
        """
        super(WarmUpCosineHardRestarts, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
        self.end_learning_rate = end_learning_rate
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name) as name:
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.initial_learning_rate, name='initial_learning_rate')
            dtype = initial_learning_rate.dtype

            end_learning_rate = math_ops.cast(self.end_learning_rate, dtype)

            global_step_recomp = math_ops.cast(step, dtype)
            cycle_steps_recomp = math_ops.cast(self.cycle_steps, dtype)
            warmup_steps_recomp = math_ops.cast(self.warmup_steps, dtype)

            return control_flow_ops.cond(
                math_ops.less(global_step_recomp, warmup_steps_recomp),
                lambda: math_ops.add(
                    end_learning_rate,
                    math_ops.multiply(
                        math_ops.divide(
                            math_ops.subtract(initial_learning_rate, end_learning_rate),
                            warmup_steps_recomp
                        ),
                        global_step_recomp
                    ),
                    name=name
                ),
                lambda: math_ops.add(
                    end_learning_rate,
                    math_ops.multiply(
                        math_ops.multiply(0.5, math_ops.subtract(initial_learning_rate, end_learning_rate)),
                        math_ops.add(
                            tf.constant(1.0),
                            math_ops.cos(
                                math_ops.multiply(
                                    tf.constant(math.pi),
                                    math_ops.divide(
                                        math_ops.mod(global_step_recomp, cycle_steps_recomp),
                                        cycle_steps_recomp
                                    )
                                )
                            )
                        )
                    ),
                    name=name
                )
            )

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'cycle_steps': self.cycle_steps,
            'warmup_steps': self.warmup_steps,
            'end_learning_rate': self.end_learning_rate,
            'name': self.name
        }


def create_optimizer(init_lr, end_lr, cycle_steps, warmup_steps):
    lr_schedule = WarmUpCosineHardRestarts(
        initial_learning_rate=init_lr,
        cycle_steps=cycle_steps,
        warmup_steps=warmup_steps,
        end_learning_rate=end_lr
    )

    optimizer = AdamWeightDecay(
        learning_rate=lr_schedule,
        weight_decay_rate=1e-2,
        epsilon=1e-6,
        exclude_from_weight_decay=['layer_norm', 'bias']
    )

    return optimizer
