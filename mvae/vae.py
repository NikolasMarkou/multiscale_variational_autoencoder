import abc
import keras

# ==============================================================================


class VAE(abc.ABC):
    """
    Describes the interface that a Variation Autoencoder must comply too
    """

    @abc.abstractmethod
    def sample(self):
        """
        From z_dim input produce an input_dim output
        :return:
        """
        raise NotImplemented()

    @abc.abstractmethod
    def predict(self):
        """
        From input_dim input produce an input_dim output
        :return:
        """
        raise NotImplemented()

    @abc.abstractmethod
    def encode(self):
        """
        From input_dim input produce an z_dim output
        :return:
        """
        raise NotImplemented()

    @property
    @abc.abstractmethod
    def z_dim(self) -> int:
        """
        Returns the size of the latent z space dimensions
        :return:
        """
        raise NotImplemented()

    @property
    @abc.abstractmethod
    def input_dim(self) -> (int,):
        """
        Returns the input dimensions shape
        :return:
        """
        raise NotImplemented()

    @property
    @abc.abstractmethod
    def model_sample(self) -> keras.models.Model:
        """
        Returns the model that can be used for sampling the latent z space and
        return a initial shape
        :return:
        """
        raise NotImplemented()

    @property
    @abc.abstractmethod
    def model_predict(self) -> keras.models.Model:
        """
        Returns the model that can take an input
        and project it into the latent z space
        :return:
        """
        raise NotImplemented()

    @property
    @abc.abstractmethod
    def model_trainable(self) -> keras.models.Model:
        """
        Returns the model that can be trained end to end
        :return:
        """
        raise NotImplemented()


# ==============================================================================
