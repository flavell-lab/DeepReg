import os
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

from deepreg import log
from deepreg.loss.label import compute_centroid_distance
from deepreg.model import layer, layer_util
from deepreg.model.backbone import GlobalNet
from deepreg.registry import REGISTRY

logger = log.get(__name__)


def dict_without(d: dict, key) -> dict:
    """
    Return a copy of the given dict without a certain key.

    :param d: dict to be copied.
    :param key: key to be removed.
    :return: the copy without a key
    """
    copied = deepcopy(d)
    copied.pop(key)
    return copied


class RegistrationModel(tf.keras.Model):
    """Interface for registration model."""

    def __init__(
        self,
        moving_image_size: Tuple,
        fixed_image_size: Tuple,
        index_size: int,
        labeled: bool,
        batch_size: int,
        config: dict,
        name: str = "RegistrationModel",
    ):
        """
        Init.

        :param moving_image_size: (m_dim1, m_dim2, m_dim3)
        :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
        :param index_size: number of indices for identify each sample
        :param labeled: if the data is labeled
        :param batch_size: total number of samples consumed per step, over all devices.
            When using multiple devices, TensorFlow automatically split the tensors.
            Therefore, input shapes should be defined over batch_size.
        :param config: config for method, backbone, and loss.
        :param name: name of the model
        """
        super().__init__(name=name)
        self.moving_image_size = moving_image_size
        self.fixed_image_size = fixed_image_size
        self.index_size = index_size
        self.labeled = labeled
        self.config = config
        self.batch_size = batch_size

        self._inputs = None  # save inputs of self._model as dict
        self._outputs = None  # save outputs of self._model as dict

        self.grid_ref = layer_util.get_reference_grid(grid_size=fixed_image_size)[
            None, ...
        ]
        self._model: tf.keras.Model = self.build_model()
        self.build_loss()

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        return dict(
            moving_image_size=self.moving_image_size,
            fixed_image_size=self.fixed_image_size,
            index_size=self.index_size,
            labeled=self.labeled,
            batch_size=self.batch_size,
            config=self.config,
            name=self.name,
        )

    @abstractmethod
    def build_model(self):
        """Build the model to be saved as self._model."""

    def build_inputs(self) -> Dict[str, tf.keras.layers.Input]:
        """
        Build input tensors.

        :return: dict of inputs.
        """
        # (batch, m_dim1, m_dim2, m_dim3)
        moving_image = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
            name="moving_image",
        )
        # (batch, f_dim1, f_dim2, f_dim3)
        fixed_image = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
            name="fixed_image",
        )
        # (batch, index_size)
        indices = tf.keras.Input(
            shape=(self.index_size,),
            batch_size=self.batch_size,
            name="indices",
        )

        if not self.labeled:
            return dict(
                moving_image=moving_image, fixed_image=fixed_image, indices=indices
            )

        # (batch, m_dim1, m_dim2, m_dim3)
        moving_label = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
            name="moving_label",
            dtype=tf.int32
        )
        # (batch, m_dim1, m_dim2, m_dim3)
        fixed_label = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
            name="fixed_label",
            dtype=tf.int32
        )
        return dict(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_label=moving_label,
            fixed_label=fixed_label,
            indices=indices,
        )

    def concat_images(
        self,
        moving_image: tf.Tensor,
        fixed_image: tf.Tensor,
        moving_label: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Adjust image shape and concatenate them together.

        :param moving_image: registration source
        :param fixed_image: registration target
        :param moving_label: optional, only used for conditional model.
        :return:
        """
        images = []

        resize_layer = layer.Resize3d(shape=self.fixed_image_size)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_image = tf.expand_dims(moving_image, axis=4)
        moving_image = resize_layer(moving_image)
        images.append(moving_image)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        fixed_image = tf.expand_dims(fixed_image, axis=4)
        images.append(fixed_image)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        if moving_label is not None:
            moving_label = tf.expand_dims(moving_label, axis=4)
            moving_label = resize_layer(moving_label)
            images.append(moving_label)

        # (batch, f_dim1, f_dim2, f_dim3, 2 or 3)
        images = tf.concat(images, axis=4)
        return images

    # TODO: add the hybrid option
    def _build_loss(self, name: str, inputs_dict: dict):
        """
        Build and add one weighted loss together with the metrics.

        :param name: name of loss, image / label / regularization.
        :param inputs_dict: inputs for loss function
        """

        if name not in self.config["loss"]:
            # loss config is not defined
            logger.warning(
                f"The configuration for loss {name} is not defined. "
                f"Therefore it is not used."
            )
            return

        loss_configs = self.config["loss"][name]
        if not isinstance(loss_configs, list):
            loss_configs = [loss_configs]

        for loss_config in loss_configs:

            if "weight" not in loss_config:
                # default loss weight 1
                logger.warning(
                    f"The weight for loss {name} is not defined."
                    f"Default weight = 1.0 is used."
                )
                loss_config["weight"] = 1.0

            # build loss
            weight = loss_config["weight"]

            if weight == 0:
                logger.warning(
                    f"The weight for loss {name} is zero." f"Loss is not used."
                )
                return

            # do not perform reduction over batch axis for supporting multi-device
            # training, model.fit() will average over global batch size automatically
            loss_layer: tf.keras.layers.Layer = REGISTRY.build_loss(
                config=dict_without(d=loss_config, key="weight"),
                default_args={"reduction": tf.keras.losses.Reduction.NONE},
            )
            loss_value = loss_layer(**inputs_dict)

            ### TODO: changed how loss is weighted
            if name == "regularization":
                hybrid = loss_config["hybrid"]
                if hybrid:
                    weighted_loss = loss_value
                else:
                    weighted_loss = loss_value * weight
            # alternative case: weighted image loss
            else:
                weighted_loss = loss_value * weight

            # add loss
            self._model.add_loss(weighted_loss)

            # add metric
            self._model.add_metric(
                loss_value, name=f"loss/{name}_{loss_layer.name}", aggregation="mean"
            )
            self._model.add_metric(
                weighted_loss,
                name=f"loss/{name}_{loss_layer.name}_weighted",
                aggregation="mean",
            )

    @abstractmethod
    def build_loss(self):
        """Build losses according to configs."""

        # input metrics
        fixed_image = self._inputs["fixed_image"]
        moving_image = self._inputs["moving_image"]
        self.log_tensor_stats(tensor=moving_image, name="moving_image")
        self.log_tensor_stats(tensor=fixed_image, name="fixed_image")

        # image loss, conditional model does not have this
        if "pred_fixed_image" in self._outputs:
            pred_fixed_image = self._outputs["pred_fixed_image"]
            self._build_loss(
                name="image",
                inputs_dict=dict(y_true=fixed_image, y_pred=pred_fixed_image),
            )

        if self.labeled:
            # input metrics
            fixed_label = self._inputs["fixed_label"]
            moving_label = self._inputs["moving_label"]
            self.log_tensor_stats(tensor=moving_label, name="moving_label")
            self.log_tensor_stats(tensor=fixed_label, name="fixed_label")

            # label loss
            pred_fixed_label = self._outputs["pred_fixed_label"]
            self._build_loss(
                name="label",
                inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
            )
            # TODO: Fix this metric with the weird partially-one-hot labeling
            # scheme
            # additional label metrics
            # tre = compute_centroid_distance(
            #    y_true=fixed_label, y_pred=pred_fixed_label, grid=self.grid_ref
            # )
            # self._model.add_metric(tre, name="metric/TRE", aggregation="mean")

    def call(
        self, inputs: Dict[str, tf.Tensor], training=None, mask=None
    ) -> Dict[str, tf.Tensor]:
        """
        Call the self._model.

        :param inputs: a dict of tensors.
        :param training: training or not.
        :param mask: maks for inputs.
        :return:
        """
        return self._model(inputs, training=training, mask=mask)  # pragma: no cover

    @abstractmethod
    def postprocess(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, Dict]:
        """
        Return a dict used for saving inputs and outputs.

        :param inputs: dict of model inputs
        :param outputs: dict of model outputs
        :return: tuple, indices and a dict.
            In the dict, each value is (tensor, normalize, on_label), where
            - normalize = True if the tensor need to be normalized to [0, 1]
            - on_label = True if the tensor depends on label
        """

    def plot_model(self, output_dir: str):
        """
        Save model structure in png.

        :param output_dir: path to the output dir.
        """
        self._model.summary(print_fn=logger.debug)
        try:
            tf.keras.utils.plot_model(
                self._model,
                to_file=os.path.join(output_dir, f"{self.name}.png"),
                dpi=96,
                show_shapes=True,
                show_layer_names=True,
                expand_nested=False,
            )
        except ImportError as err:  # pragma: no cover
            logger.error(
                "Failed to plot model structure. "
                "Please check if graphviz is installed. "
                "Error message is: %s.",
                err,
            )

    def log_tensor_stats(self, tensor: tf.Tensor, name: str):
        """
        Log statistics of a given tensor.

        :param tensor: tensor to monitor.
        :param name: name of the tensor.
        """
        flatten = tf.cast(tf.reshape(tensor, shape=(self.batch_size, -1)),
                tf.float32)
        self._model.add_metric(
            tf.reduce_mean(flatten, axis=1),
            name=f"metric/{name}_mean",
            aggregation="mean",
        )
        self._model.add_metric(
            tf.reduce_min(flatten, axis=1),
            name=f"metric/{name}_min",
            aggregation="min",
        )
        self._model.add_metric(
            tf.reduce_max(flatten, axis=1),
            name=f"metric/{name}_max",
            aggregation="max",
        )
def save_data(loss, moving_image, fixed_image, moving_label, fixed_label, pred_fixed_image, pred_fixed_label):
    if not tf.reduce_any(tf.math.is_nan(loss)):
        return None
    save_idx = np.random.randint(1000000)
    with h5py.File(os.path.join("/data1/prj_register/test_save_fit", f'train_{save_idx}.h5'), "w") as f:
        f["moving_image"]  = moving_image.numpy()
        f["fixed_image"] = fixed_image.numpy()
        f["moving_label"] = moving_label.numpy()
        f["fixed_label"] = fixed_label.numpy()
        f["pred_fixed_image"] = pred_fixed_image.numpy()
        f["pred_fixed_label"] = pred_fixed_label.numpy()
    raise ValueError("NaN loss")

@REGISTRY.register_model(name="ddf")
class DDFModel(RegistrationModel):
    """
    A registration model predicts DDF.

    When using global net as backbone,
    the model predicts an affine transformation parameters,
    and a DDF is calculated based on that.
    """

    name = "DDFModel"

    def __init__(
        self,
        moving_image_size: Tuple,
        fixed_image_size: Tuple,
        moving_label_size: Tuple,
        fixed_label_size: Tuple,
        index_size: int,
        labeled: bool,
        batch_size: int,
        config: dict,
        name: str = "DDFModel",
    ):
        """
        Init.

        :param moving_image_size: (m_dim1, m_dim2, m_dim3)
        :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
        :param moving_label_size: (m_dim1, m_dim2, m_dim3)
        :param fixed_label_size: (f_dim1, f_dim2, f_dim3)
        :param index_size: number of indices for identify each sample
        :param labeled: if the data is labeled
        :param batch_size: total number of samples consumed per step, over all devices.
            When using multiple devices, TensorFlow automatically split the tensors.
            Therefore, input shapes should be defined over batch_size.
        :param config: config for method, backbone, and loss.
        :param name: name of the model
        """
        self.fixed_label_size=fixed_label_size
        self.moving_label_size=moving_label_size
        super().__init__(
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            index_size=index_size,
            labeled=labeled,
            batch_size=batch_size,
            config=config,
            name=name,
        )

    
    def train_step(self, data):
        """The logic for one training step.

        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happens in fit](
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.

        This method should contain the mathematical logic for one step of
        training.  This typically includes the forward pass, loss calculation,
        backpropagation, and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function`
        and `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)


        tf.py_function(save_data,
               [loss, x['moving_image'], x['fixed_image'], x['moving_label'], x['fixed_label'],
                y_pred['pred_fixed_image'], y_pred['pred_fixed_label']],
               Tout=[])

        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def build_inputs(self) -> Dict[str, tf.keras.layers.Input]:
        """
        Build input tensors.

        :return: dict of inputs.
        """
        # (batch, m_dim1, m_dim2, m_dim3)
        moving_image = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
            name="moving_image",
        )
        # (batch, f_dim1, f_dim2, f_dim3)
        fixed_image = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
            name="fixed_image",
        )
        # (batch, index_size)
        indices = tf.keras.Input(
            shape=(self.index_size,),
            batch_size=self.batch_size,
            name="indices",
        )

        if not self.labeled:
            return dict(
                moving_image=moving_image, fixed_image=fixed_image, indices=indices
            )

        # (batch, m_dim1, m_dim2, m_dim3)
        moving_label = tf.keras.Input(
            shape=self.moving_label_size,
            batch_size=self.batch_size,
            name="moving_label",
            dtype=tf.int32
        )
        # (batch, m_dim1, m_dim2, m_dim3)
        fixed_label = tf.keras.Input(
            shape=self.fixed_label_size,
            batch_size=self.batch_size,
            name="fixed_label",
            dtype=tf.int32
        )
        return dict(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_label=moving_label,
            fixed_label=fixed_label,
            indices=indices,
        )

    def _resize_interpolate(self, field, control_points):
        resize = layer.ResizeCPTransform(control_points)
        field = resize(field)

        interpolate = layer.BSplines3DTransform(control_points, self.fixed_image_size)
        field = interpolate(field)

        return field

    def build_model(self):
        """Build the model to be saved as self._model."""
        # build inputs
        self._inputs = self.build_inputs()
        moving_image = self._inputs["moving_image"]  # (batch, m_dim1, m_dim2, m_dim3)
        fixed_image = self._inputs["fixed_image"]  # (batch, f_dim1, f_dim2, f_dim3)

        # build ddf
        control_points = self.config["backbone"].pop("control_points", False)
        backbone_inputs = self.concat_images(moving_image, fixed_image)
        backbone = REGISTRY.build_backbone(
            config=self.config["backbone"],
            default_args=dict(
                image_size=self.fixed_image_size,
                out_channels=3,
                out_kernel_initializer="zeros",
                out_activation=None,
            ),
        )

        if isinstance(backbone, GlobalNet):
            # (f_dim1, f_dim2, f_dim3, 3), (4, 3)
            ddf, theta = backbone(inputs=backbone_inputs)
            self._outputs = dict(ddf=ddf, theta=theta)
        else:
            # (f_dim1, f_dim2, f_dim3, 3)
            ddf = backbone(inputs=backbone_inputs)
            ddf = (
                self._resize_interpolate(ddf, control_points) if control_points else ddf
            )
            self._outputs = dict(ddf=ddf)

        # build outputs
        warping = layer.Warping(fixed_image_size=self.fixed_image_size)
        # (f_dim1, f_dim2, f_dim3)
        pred_fixed_image = warping(inputs=[ddf, moving_image])
        self._outputs["pred_fixed_image"] = pred_fixed_image

        if not self.labeled:
            return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

        warping_centroids = layer.CentroidWarping(fixed_image_size=self.fixed_image_size)
        # warping_multichannel = layer.MultiChannelWarping(fixed_image_size=self.fixed_image_size)
        # (f_dim1, f_dim2, f_dim3)
        # TODO: branch whether "moving" label should be moving image or fixed centroids based off of the label loss function, and modify loss function calling appropriately
        moving_label = self._inputs["moving_label"] # for centroid inputs, this will be the FIXED label
        # TODO: Put 200 in an initialization argument
        # moving_label_one_hot = tf.one_hot(moving_label, depth=50, axis=-1)
        # print("Moving label data type")
        # print(moving_label.dtype)
        # print("One-hot moving label data type and shape")
        # print(moving_label_one_hot.dtype)
        # print(moving_label_one_hot.shape)
        pred_fixed_label = warping_centroids(inputs=[ddf, moving_label])
        # print("Fixed label data type")
        # print(pred_fixed_label.dtype)
        self._outputs["pred_fixed_label"] = pred_fixed_label # for centroid inputs, this will be the predicted MOVING label
        return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

    def build_loss(self):
        """Build losses according to configs."""
        super().build_loss()

        # ddf loss and metrics
        ddf = self._outputs["ddf"]
        ### TODO: call build hybrid loss
        self._build_loss(name="regularization", inputs_dict=dict(inputs=ddf))
        self.log_tensor_stats(tensor=ddf, name="ddf")

    def postprocess(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, Dict]:
        """
        Return a dict used for saving inputs and outputs.

        :param inputs: dict of model inputs
        :param outputs: dict of model outputs
        :return: tuple, indices and a dict.
            In the dict, each value is (tensor, normalize, on_label), where
            - normalize = True if the tensor need to be normalized to [0, 1]
            - on_label = True if the tensor depends on label
        """
        indices = inputs["indices"]
        processed = dict(
            moving_image=(inputs["moving_image"], True, False),
            fixed_image=(inputs["fixed_image"], True, False),
            ddf=(outputs["ddf"], True, False),
            pred_fixed_image=(outputs["pred_fixed_image"], True, False),
        )

        # save theta for affine model
        if "theta" in outputs:
            processed["theta"] = (outputs["theta"], None, None)  # type: ignore

        if not self.labeled:
            return indices, processed

        processed = {
            **dict(
                moving_label=(inputs["moving_label"], False, True),
                fixed_label=(inputs["fixed_label"], False, True),
                pred_fixed_label=(outputs["pred_fixed_label"], False, True),
            ),
            **processed,
        }

        return indices, processed


@REGISTRY.register_model(name="dvf")
class DVFModel(DDFModel):
    """
    A registration model predicts DVF.

    DDF is calculated based on DVF.
    """

    name = "DVFModel"

    def build_model(self):
        """Build the model to be saved as self._model."""
        # build inputs
        self._inputs = self.build_inputs()
        moving_image = self._inputs["moving_image"]
        fixed_image = self._inputs["fixed_image"]
        control_points = self.config["backbone"].pop("control_points", False)

        # build ddf
        backbone_inputs = self.concat_images(moving_image, fixed_image)
        backbone = REGISTRY.build_backbone(
            config=self.config["backbone"],
            default_args=dict(
                image_size=self.fixed_image_size,
                out_channels=3,
                out_kernel_initializer="zeros",
                out_activation=None,
            ),
        )
        dvf = backbone(inputs=backbone_inputs)
        dvf = self._resize_interpolate(dvf, control_points) if control_points else dvf
        ddf = layer.IntDVF(fixed_image_size=self.fixed_image_size)(dvf)

        # build outputs
        self._warping = layer.Warping(fixed_image_size=self.fixed_image_size)
        # (f_dim1, f_dim2, f_dim3, 3)
        pred_fixed_image = self._warping(inputs=[ddf, moving_image])

        self._outputs = dict(dvf=dvf, ddf=ddf, pred_fixed_image=pred_fixed_image)

        if not self.labeled:
            return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

        # (f_dim1, f_dim2, f_dim3, 3)
        moving_label = self._inputs["moving_label"]
        pred_fixed_label = self._warping(inputs=[ddf, moving_label])

        self._outputs["pred_fixed_label"] = pred_fixed_label
        return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

    def build_loss(self):
        """Build losses according to configs."""
        super().build_loss()

        # dvf metrics
        dvf = self._outputs["dvf"]
        self.log_tensor_stats(tensor=dvf, name="dvf")

    def postprocess(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, Dict]:
        """
        Return a dict used for saving inputs and outputs.

        :param inputs: dict of model inputs
        :param outputs: dict of model outputs
        :return: tuple, indices and a dict.
            In the dict, each value is (tensor, normalize, on_label), where
            - normalize = True if the tensor need to be normalized to [0, 1]
            - on_label = True if the tensor depends on label
        """
        indices, processed = super().postprocess(inputs=inputs, outputs=outputs)
        processed["dvf"] = (outputs["dvf"], True, False)
        return indices, processed


@REGISTRY.register_model(name="conditional")
class ConditionalModel(RegistrationModel):
    """
    A registration model predicts fixed image label without DDF or DVF.
    """

    name = "ConditionalModel"

    def build_model(self):
        """Build the model to be saved as self._model."""
        assert self.labeled

        # build inputs
        self._inputs = self.build_inputs()
        moving_image = self._inputs["moving_image"]
        fixed_image = self._inputs["fixed_image"]
        moving_label = self._inputs["moving_label"]

        # build ddf
        backbone_inputs = self.concat_images(moving_image, fixed_image, moving_label)
        backbone = REGISTRY.build_backbone(
            config=self.config["backbone"],
            default_args=dict(
                image_size=self.fixed_image_size,
                out_channels=1,
                out_kernel_initializer="glorot_uniform",
                out_activation="sigmoid",
            ),
        )
        # (batch, f_dim1, f_dim2, f_dim3)
        pred_fixed_label = backbone(inputs=backbone_inputs)
        pred_fixed_label = tf.squeeze(pred_fixed_label, axis=4)

        self._outputs = dict(pred_fixed_label=pred_fixed_label)
        return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

    def postprocess(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, Dict]:
        """
        Return a dict used for saving inputs and outputs.

        :param inputs: dict of model inputs
        :param outputs: dict of model outputs
        :return: tuple, indices and a dict.
            In the dict, each value is (tensor, normalize, on_label), where
            - normalize = True if the tensor need to be normalized to [0, 1]
            - on_label = True if the tensor depends on label
        """
        indices = inputs["indices"]
        processed = dict(
            moving_image=(inputs["moving_image"], True, False),
            fixed_image=(inputs["fixed_image"], True, False),
            pred_fixed_label=(outputs["pred_fixed_label"], True, True),
            moving_label=(inputs["moving_label"], False, True),
            fixed_label=(inputs["fixed_label"], False, True),
        )

        return indices, processed
