function [a_vec, output] = forward_propagation (data, weights_1, weights_2, units_hidden_layer)
  x = [ 1 data ]';
   z = weights_1 * x;
  a_vec = 1 ./ (1 + (e .^ -z));
  output = weights_2 * a_vec;
endfunction