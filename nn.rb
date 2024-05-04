require 'csv'

""" sigmoid - it sucks
def f(x)
    return (1 / (1 + Math.exp(-x))).to_f
end

def df(x)
    return x * (1 - x)
end
"""

def f(x)
    return x > 0 ? x : 0.3 * x
end

def df(x)
    return x > 0 ? 1 : 0.3
end

def argmax(arr)
    max = 0
    for i in 1..arr.length-1 do
        if arr[max] < arr[i]
            max = i
        end
    end
    return max
end

class Neuron
    attr_accessor :b, :a, :z, :e

    def initialize(b, a, z)
        @b = b
        @a = a
        @z = z
        @e = 0 

    end
end

class Network
    attr_accessor :w, :neurons, :layer_sizes, :lr

    def initialize(layer_sizes, lr=1)
        @neurons = Array.new(layer_sizes.length) do |l| 
            init_layer(layer_sizes[l]) 
        end
        @w = Array.new(layer_sizes.length-1)  do |l| 
            Array.new(layer_sizes[l+1]) {Array.new(layer_sizes[l]) {0}}
        end
        @layer_sizes = layer_sizes
        @lr = lr
        @cumulative_delta = 0
    end

    def init_layer(layer_size)
        return Array.new(layer_size) {Neuron.new(0, 0, 0) } 
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
            output.e = (output.a - expected[o_id]) * df(output.a) 
        end

        #now just send the error back
        for layer in (@neurons.length-1).downto(1) do 
            @neurons[layer-1].each_with_index do |target, t_id|
                target.e = 0
                @neurons[layer].each_with_index do |source, s_id|
                    target.e += @w[layer-1][s_id][t_id] * source.e
                end
                target.e *= df(target.a) 
            end
        end
        
        for layer in 0..@neurons.length-2 do
            @neurons[layer].each_with_index do |source, s_id|
                @neurons[layer+1].each_with_index do |target, t_id|
                    @cumulative_delta += (@lr * target.e).abs()
                    @w[layer][t_id][s_id] -= @lr * target.e * source.a
                    source.b -= @lr * target.e
                end
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
            #map neuron activations to array
            out = @neurons[-1].map {|n| n.a}
            if argmax(outputs[i]) == argmax(out) 
                correct_count += 1
            end
        end
        percentage = correct_count.to_f / inputs.length * 100
        p "Accuracy: %0.2f " % [percentage] 
        p "Collective weight change: %0.2f" % [@cumulative_delta] 
        p "learning rate: %0.2f" % [@lr]
    end
end

def split_data(x, y, percentage: 0.2, shuffle: false, seed: nil)
    training_x, training_y = [], []
    testing_x, testing_y = [], []
    n_train = ((1 - percentage) * (x.length - 1)).floor
    n_test = x.length-n_train
    if shuffle == true
        seed = seed.nil? ? (Time.now.to_i * rand()).to_i : seed
        r = Random.new(seed)
        for i in (x.length-1).downto(1)
            el = ((i+1) * r.rand()).floor()
            x[i], x[el] = x[el], x[i]
            y[i], y[el] = y[el], y[i]
        end
    end

    for row in 0..n_train do
        training_x.append(x[row])
        training_y.append(y[row])
    end

    for row in n_train+1..x.length-1 do
       testing_x.append(x[row])
       testing_y.append(y[row])
    end
    return training_x, training_y, testing_x, testing_y
end

