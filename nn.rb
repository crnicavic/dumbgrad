#neural network proof of concept + i can play with it

#weights go to next neuron

def f(x)
    return 1 / (1 + x.abs())
end

class Neuron
    attr_accessor :b, :a, :z, :w

    def initialize(b, w, a, z)
        @b = b
        @w = w
        @a = a
        @z = z
    end
end

class Layer
    attr_accessor :neurons

    def initialize(size, nl_size=nil)
        if nl_size.nil?
            @neurons = Array.new(size) {Neuron.new(Random.rand(), Array.new(), 0, 0)}
            return
        end
        arr = Array.new(nl_size) {Random.rand()}
        @neurons = Array.new(size) {Neuron.new(Random.rand(), arr, 0, 0)}
    end

    #calculate the activations for the next layer
    def propagate(layer)
        for nl_neuron in layer.neurons
            nl_neuron.a = 0
        end

        for neuron in @neurons
            for nl_id in 0..layer.neurons.length-1
                layer.neurons[nl_id].a += neuron.w[nl_id] * neuron.z
            end
        end

        for nl_neuron in layer.neurons
            nl_neuron.z = f(nl_neuron.a)
        end

    end
end


class NeuralNetwork
    attr_accessor :layers

    def initialize(layer_sizes)
        @layers = Array.new()
        for l in 0..layer_sizes.length-1 do
            @layers[l] = Layer.new(layer_sizes[l], layer_sizes[l+1])
        end
    end


    def feedforward(input)
        for n_id in 0..@layers[0].neurons.length-1
            @layers[0].neurons[n_id].z = f(input[n_id])
        end

        #ruby ranges include last member
        for l_id in 0..@layers.length-2
            @layers[l_id].propagate(@layers[l_id+1])
        end
    end
end


network = NeuralNetwork.new([5, 6, 12, 3])
network.feedforward([0, 1, 4, 6, 9])
puts network.layers[-1].neurons.inspect
