require '../../nn.rb'

# range of output values in the data
diagnosis_count = 5

# formatting y so that it mimics the wanted output per epoch
# y[0] = [0, 0, 0, 1, 0] for example - output for first epoch 
data = CSV.read("heart.csv", converters: :numeric)
x = data.map{|row| row[0..-2]}
y = Array.new(x.length) {Array.new(diagnosis_count) {0}}

# map the last column of the csv to an array, it's value
# represents which neuron i want to be the most active
(data.map {|row| row[-1]}).each_with_index do |output, o_id|
    y[o_id][output] = 1 
end

training_x, training_y, testing_x, testing_y = split_data(x, y, shuffle: true)
iln = training_x[0].length
net = Network.new([iln, iln*2, iln*3, iln*2, iln, diagnosis_count], lr=0.01)

net.train(training_x, training_y)
net.test(testing_x, testing_y)
