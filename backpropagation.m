function [new_weights_h, new_weights_o, error_vec] = backpropagation (data, weights_1, weights_2, units_hidden_layer, alpha)
  warning('off');
  new_weights_h = weights_1;
  new_weights_o = weights_2;
  error_vec = [];
  iter = 0;
  max_iters = 10000;

  while iter < max_iters
    printf('\r%d%s',(iter * 100)/ max_iters, '%');
    acumulated_error = 0;
    iter += 1;
    for i = 1:length(data)

      [a_vec, output] = forward_propagation(data(i,1:2), new_weights_h, new_weights_o, units_hidden_layer);

      transformed_data = data(i,3);
      error = output - transformed_data;
      acumulated_error += sum(abs(error));
      new_weights_o = new_weights_o - (alpha .* error .* a_vec');
      new_weights_h = new_weights_h - (alpha .* error .* new_weights_o' .* a_vec .* (1 - a_vec) .* [1 data(i,1:2)]);

    endfor
    error_vec = [error_vec; acumulated_error];

    if iter > 1 && (error_vec(iter - 1) > acumulated_error) ...
      && (((error_vec(iter - 1) - acumulated_error) <= 0.0001));
      disp(sprintf('\rEarly exit by error convergence with %d iterations.', iter));
      break
    endif

  endwhile

endfunction
