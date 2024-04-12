require 'csv'

def f(x)
    return 1 / (1 + Math.exp(-x))
end

#instead of loading some data, just do a simple sine until shit works
def test_function(x)
    return Math.sin(x)
end



class Neuron
    attr_accessor :b, :a, :z, :e

    def initialize(b, a, z)
        @b = b
        @a = a
        @z = z
        @e = 0      #really lazy

    end
end

class Network
    attr_accessor :w, :neurons, :layer_sizes, :lr

    def initialize(layer_sizes, lr=0.1)
      # TODO: Create methods or functions to make this prettier and SHORTER
        @neurons = Array.new(layer_sizes.length) {|l| Array.new(layer_sizes[l]) {Neuron.new(0, 0, 0) } } 
        @w = Array.new(layer_sizes.length-1) {|l| Array.new(layer_sizes[l+1]) {Array.new(layer_sizes[l]) {rand() * 10 - 5}}}
        @layer_sizes = layer_sizes
        @lr = lr
    end

    def feedforward(input)
        @neurons[0].each_with_index do |neuron, n_id|
            neuron.z = input[n_id]
            neuron.a = f(neuron.z)
        end

        #ranges in ruby include last number ffs
        for layer in 1..@neurons.length-1
            @neurons[layer].each_with_index do |target, t_id|
                target.z = 0
                @neurons[layer-1].each_with_index do |source, s_id|
                    target.z += @w[layer-1][t_id][s_id] * source.a 
                end
                target.z += target.b
                target.a = f(target.z)
            end
        end
    end
    
    def backprop(expected)
        @neurons[-1].each_with_index do |output, o_id|
            output.e = expected[o_id] - output.a 
        end
        #now just send the error back
        for layer in (@neurons.length-1).downto(1) do 
            #this time it goes backwards source is the target, and vice versa
            @neurons[layer-1].each_with_index do |target, t_id|
                target.e = 0
                @neurons[layer].each_with_index do |source, s_id|
                    target.e += @w[layer-1][s_id][t_id] * source.e
                    #used weight is no longer relevant, i can update it
                    @w[layer-1][s_id][t_id] -= source.e * target.a * @lr * (-1)
                end
                target.e *= target.a * (1 - target.a) 
            end
        end
    end

    def train(data, expected_data)
    end
end

#this will be fixed to split outputs but currently 
#not necessary
def split_inputs(inputs)
    training_inputs = []
    testing_inputs = []
    inputs.each do |d|
        if rand() < 0.2
            testing_inputs.append(d)
        else
            training_inputs.append(d)
        end
    end
    return testing_inputs, training_inputs
end

data = (0..6.28).step(0.1).to_a
testing_inputs, training_inputs = split_inputs(data)
training_outputs = Array.new(training_inputs.length) {|d| Math.cos(d)}
testing_outputs = Array.new(testing_inputs.length) {|d| Math.cos(d)}
net = Network.new([1, 5, 6, 1]) 

#training
for i in 0..training_inputs.length-1 do
    net.feedforward([training_inputs[i]])
    net.backprop([training_outputs[i]])
end

#test
correct_count = 0
for i in 0..testing_inputs.length-1 do
    net.feedforward([testing_inputs[i]])
    err = net.neurons[-1][0].a - testing_outputs[i]
    if err.abs() < 0.1
        correct_count += 1
    end
end

percentage = correct_count.to_f / testing_inputs.length * 100
puts percentage
