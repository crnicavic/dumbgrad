#neural network proof of concept + i can play with it

#weights go to next neuron
class Neuron
    attr_reader :b, :a, :z, :w

    def initialize(b, w, a, z)
        @b = b
        @w = w
        @a = a
        @z = z
    end
end

class Layer
    attr_reader :neurons

    #nl_size size of previous layer
    def initialize(size, nl_size=nil)
        if nl_size.nil?
            @neurons = Array.new(size) {Neuron.new(Random.rand(), Array.new(), 0, 0)}
        end
        @neurons = Array.new(size) {Neuron.new(Random.rand(), Array.new(nl_size), 0, 0)}
    end

    #calculate the activations for the next layer
    def propagate(layer)
        for nl_neuron in layer.neurons
            nl_neuron.a = 0
        end

        for neuron in @neurons
            for nl_neuron in layer.neurons
                nl_neuron.a += neuron.w[nl_id] * neuron.a
            end
        end

        for nl_neuron in layer.neurons
            nl_neuron.z = nl_neuron.f(nl_neuron.a)
        end

    end
end


class NeuralNetwork
    attr_reader :layers

    def initialize(layer_sizes)
        @layers = Array.new()
        for l in 0..layer_sizes.length-1 do
          @layers[l] = Layer.new(layer_sizes[l], layer_sizes[l+1])
        end
    end

    def feedforward(input)
        for n_id in 0..@layers[0].neurons.length
            @layers[0].neurons[n_id] = input[n_id]
        end
        @layers.each_with_index {|layer, i| layer.propagate(@layers[i])}
    end
end


network = NeuralNetwork.new([5, 6, 12, 3])
network.feedforward([0, 1, 4, 6, 9])
puts *network.layers[1].neurons[2].b
