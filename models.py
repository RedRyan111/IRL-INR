import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)


class Encoder(nn.Module):
    def __init__(self, n, latent_dim, hidden_dim=256, num_layers=3, activation=nn.Tanh, resid=False):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.n = n

        layers = [nn.Linear(n, hidden_dim),
                  activation(),
                  ]
        for _ in range(1, num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())

        # layers.append(nn.Linear(hidden_dim, 2*latent_dim))
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        # x is (batch,num_coords)
        images = images.view(images.shape[0], -1)
        z = self.layers(images)
        return z


class FunctionRepresentation(nn.Module):
    """Function to represent a single datapoint. For example this could be a
    function that takes pixel coordinates as input and returns RGB values, i.e.
    f(x, y) = (r, g, b).
    Args:
        coordinate_dim (int): Dimension of input (coordinates).
        feature_dim (int): Dimension of output (features).
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        encoding (torch.nn.Module): Encoding layer, usually one of
            Identity or FourierFeatures.
        final_non_linearity (torch.nn.Module): Final non linearity to use.
            Usually nn.Sigmoid() or nn.Tanh().
    """

    def __init__(self, feature_dim, layer_sizes, num_frequencies=128):
        super(FunctionRepresentation, self).__init__()
        self.feature_dim = feature_dim  # 3 or 1
        self.layer_sizes = layer_sizes
        self.non_linearity = nn.ReLU()
        self.num_frequencies = num_frequencies
        self.final_non_linearity = nn.Sigmoid()

        self._init_neural_net()

    def _init_neural_net(self):
        """
        """
        # First layer transforms coordinates into a positional encoding
        # Check output dimension of positional encoding
        prev_num_units = self.num_frequencies * 2

        # Build MLP layers
        forward_layers = []
        for num_units in self.layer_sizes:
            forward_layers.append(nn.Linear(prev_num_units, num_units))
            forward_layers.append(self.non_linearity)
            prev_num_units = num_units
        forward_layers.append(nn.Linear(prev_num_units, self.feature_dim))
        forward_layers.append(self.final_non_linearity)
        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, coordinates, weights, biases):
        """Forward pass. Given a set of coordinates, returns feature at every
        coordinate.
        Args:
            coordinates (torch.Tensor): Shape (batch_size, coordinate_dim)
        """
        # print('coordinates.shape: ', coordinates.shape)
        features = coordinates
        for i in range(len(weights)):
            features = F.linear(features, weights[i], biases[i])
            if i == len(weights) - 1:
                features = self.final_non_linearity(features)
            else:
                features = self.non_linearity(features)
        features = features.squeeze()
        return features

    def get_weight_shapes(self):
        """Returns lists of shapes of weights and biases in the network."""
        weight_shapes = []
        bias_shapes = []
        for param in self.forward_layers.parameters():
            if len(param.shape) == 1:
                bias_shapes.append(param.shape)
            if len(param.shape) == 2:
                weight_shapes.append(param.shape)
        return weight_shapes, bias_shapes

    def get_weights_and_biases(self):
        """Returns list of weights and biases in the network."""
        weights = []
        biases = []
        for param in self.forward_layers.parameters():
            if len(param.shape) == 1:
                biases.append(param)
            if len(param.shape) == 2:
                weights.append(param)
        return weights, biases

    def set_weights_and_biases(self, weights, biases):
        """Sets weights and biases in the network.
        Args:
            weights (list of torch.Tensor):
            biases (list of torch.Tensor):
        Notes:
            The inputs to this function should have the same form as the outputs
            of self.get_weights_and_biases.
        """
        weight_idx = 0
        bias_idx = 0
        with torch.no_grad():
            for param in self.forward_layers.parameters():
                if len(param.shape) == 1:
                    param.copy_(biases[bias_idx])
                    bias_idx += 1
                if len(param.shape) == 2:
                    param.copy_(weights[weight_idx])
                    weight_idx += 1

    def duplicate(self):
        """Returns a FunctionRepresentation instance with random weights."""
        # Extract device
        device = next(self.parameters()).device
        # Create new function representation and put it on same device
        return FunctionRepresentation(self.feature_dim,
                                      self.layer_sizes).to(device)


class HyperNetwork(nn.Module):
    """Hypernetwork that outputs the weights of a function representation.
    Args:
        function_representation (models.function_representation.FunctionRepresentation):
        latent_dim (int): Dimension of latent vectors.
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        non_linearity (torch.nn.Module):
    """

    def __init__(self, function_representation, latent_dim, hypernet_layer_sizes):
        super(HyperNetwork, self).__init__()
        self.function_representation = function_representation
        self.latent_dim = latent_dim
        self.layer_sizes = hypernet_layer_sizes
        self.non_linearity = nn.LeakyReLU(0.1)
        self._infer_output_shapes()
        """Initializes weights of hypernetwork."""
        forward_layers = []
        prev_num_units = self.latent_dim
        for num_units in self.layer_sizes:
            forward_layers.append(nn.Linear(prev_num_units, num_units))
            forward_layers.append(self.non_linearity)
            prev_num_units = num_units
        forward_layers.append(nn.Linear(prev_num_units, self.output_dim))
        self.forward_layers = nn.Sequential(*forward_layers)

    def _infer_output_shapes(self):
        """Uses function representation to infer correct output shapes for
        hypernetwork (i.e. so dimension matches size of weights in function
        representation) network."""
        self.weight_shapes, self.bias_shapes = self.function_representation.get_weight_shapes()
        num_layers = len(self.weight_shapes)

        # Calculate output dimension
        self.output_dim = 0
        for i in range(num_layers):
            # Add total number of weights in weight matrix
            self.output_dim += self.weight_shapes[i][0] * self.weight_shapes[i][1]
            # Add total number of weights in bias vector
            self.output_dim += self.bias_shapes[i][0]

        # Calculate partition of output of network, so that output network can
        # be reshaped into weights of the function representation network
        # Partition first part of output into weight matrices
        start_index = 0
        self.weight_partition = []
        for i in range(num_layers):
            weight_size = self.weight_shapes[i][0] * self.weight_shapes[i][1]
            self.weight_partition.append((start_index, start_index + weight_size))
            start_index += weight_size

        # Partition second part of output into bias matrices
        self.bias_partition = []
        for i in range(num_layers):
            bias_size = self.bias_shapes[i][0]
            self.bias_partition.append((start_index, start_index + bias_size))
            start_index += bias_size

        self.num_layers_function_representation = num_layers

    def output_to_weights(self, output):
        """Converts output of function distribution network into list of weights
        and biases for function representation networks.
        Args:
            output (torch.Tensor): Output of neural network as a tensor of shape
                (batch_size, self.output_dim).
        Notes:
            Each element in batch will correspond to a separate function
            representation network, therefore there will be batch_size sets of
            weights and biases.
        """
        all_weights = {}
        all_biases = {}
        # Compute weights and biases separately for each element in batch
        for i in range(output.shape[0]):
            weights = []
            biases = []
            # Add weight matrices
            for j, (start_index, end_index) in enumerate(self.weight_partition):
                weight = output[i, start_index:end_index]
                weights.append(weight.view(*self.weight_shapes[j]))
            # Add bias vectors
            for j, (start_index, end_index) in enumerate(self.bias_partition):
                bias = output[i, start_index:end_index]
                biases.append(bias.view(*self.bias_shapes[j]))
            # Add weights and biases for this function representation to batch
            all_weights[i] = weights
            all_biases[i] = biases
        return all_weights, all_biases

    def forward(self, latents):
        """Compute weights of function representations from latent vectors.
        Args:
            latents (torch.Tensor): Shape (batch_size, latent_dim).
        """
        output = self.forward_layers(latents)
        return self.output_to_weights(output)
