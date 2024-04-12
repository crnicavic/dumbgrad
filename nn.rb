require 'csv'

def f(x)
    return 1 / (1 + Math.exp(-x))
end

def argmax(arr)
    min = 0
    for i in 1..arr.length-1 do
        if arr[min] > arr[i]
            min = i
        end
    end
    return min
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
        @w = Array.new(layer_sizes.length-1) {|l| Array.new(layer_sizes[l+1]) {Array.new(layer_sizes[l]) {rand()}}}
        @layer_sizes = layer_sizes
        @lr = lr
    end

    def feedforward(input)
        @neurons[0].each_with_index do |neuron, n_id|
            neuron.z = input[n_id]
            neuron.a = f(neuron.z)
        end

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
            output.e = (expected[o_id] - output.a) * output.a * (1 - output.a) 
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

    def train(inputs, outputs)
        #training
        for i in 0..inputs.length-1 do
            self.feedforward(inputs[i])
            self.backprop(outputs[i])
        end
    end

    def test(inputs, outputs)
        correct_count = 0
        for i in 0..inputs.length-1 do
            self.feedforward(inputs[i])
            out = @neurons[-1].map { |n| n.a}
            if argmax(outputs[i]) == argmax(out)
                correct_count += 1
            end
        end
        percentage = correct_count.to_f / inputs.length * 100
        puts percentage
    end
end

def split_data(x, y, percentage=0.2)
    training_x, training_y = [], []
    testing_x, testing_y = [], []
    for row in 1..x.length-1 do
        if rand() < percentage
            testing_x.append(x[row])
            testing_y.append(y[row])
        else
            training_x.append(x[row])
            training_y.append(y[row])
        end
    end
    return training_x, training_y, testing_x, testing_y
end

data = CSV.read("heart.csv", converters: :numeric)
x = data.map{|row| row[0..-2]}
y = Array.new(x.length) {Array.new(5) {0}}

(data.map {|row| row[-1]}).each_with_index do |output, o_id|
    y[o_id][output] = 1 
end

training_x, training_y, testing_x, testing_y = split_data(x, y)
net = Network.new([training_x[0].length, 10, 10, 20, 5])

net.train(training_x, training_y)
net.test(testing_x, testing_y)

"
data = (0..6.28).step(0.05).to_a
testing_inputs, training_inputs = split_inputs(data)
training_outputs = Array.new(training_inputs.length) {|d| Math.cos(d)}
testing_outputs = Array.new(testing_inputs.length) {|d| Math.cos(d)}

net = Network.new([1, 5, 1]) 

net.train(testing_inputs, training_outputs)
net.test(testing_inputs, testing_outputs)
"
