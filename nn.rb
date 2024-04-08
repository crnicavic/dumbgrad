def f(x)
    return 1 / Math.exp(x)
end

class Neuron
    attr_accessor :b, :a, :z

    def initialize(b, a, z)
        @b = b
        @a = a
        @z = z

    end
end

class Network
    attr_accessor :w, :neurons

    def initialize(layer_sizes)
      # TODO: Create methods or functions to make this prettier
        @neurons = Array.new(layer_sizes.length) {|l| Array.new(layer_sizes[l]) {Neuron.new(rand(), 0, 0) } } 
        @w = Array.new(layer_sizes.length-1) {|l| Array.new(layer_sizes[l+1]) {Array.new(layer_sizes[l]) {rand()}}}
    end

    def feedforward(input)
        @neurons[0].each_with_index do |neuron, n_id|
            neuron.z = input[n_id]
            neuron.a = f(neuron.z)
        end

        #ranges in ruby include last number ffs
        for layer in 1..@neurons.length-1
            @neurons[layer].each_with_index do |target, t_id|
                @neurons[layer-1].each_with_index do |source, s_id|
                    target.z += @w[layer-1][t_id][s_id] * source.a 
                end
                target.a = f(target.z)
            end
        end
    end
end

net = Network.new([4, 5, 3])
net.feedforward([1, 0, 2, 4])

net.neurons[-1].each do |neuron|
    puts neuron.a
end
