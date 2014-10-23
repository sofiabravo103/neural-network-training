#!/usr/bin/octave

d1_500 = dlmread('datos_P1_2_SD2014_n500.txt');
d1_1000 = dlmread('datos_P1_2_SD2014_n1000.txt');
d1_2000 = dlmread('datos_P1_2_SD2014_n2000.txt');

d2_500 = dlmread('datos500');
d2_1000 = dlmread('datos1000');
d2_2000 = dlmread('datos2000');

training_set = d1_500;
units_hidden_layer = 5; % Important: without countung bias unit
output_units = 1;
dataset_name = sprintf('Datos del proyecto con %d puntos.',length(training_set));
%dataset_name = sprintf('Generado por el grupo con %d puntos.',length(training_set));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

weights_1 = rand(units_hidden_layer, 3);
weights_2 = rand(output_units, units_hidden_layer);


[done_weights_1, done_weights_2, error_vec] = backpropagation(training_set, ...
  weights_1, weights_2, units_hidden_layer, 0.01);

title_plot = sprintf('Conjunto de entrenamiento: %s. \nPara %d unidades en la capa oculta, culminado en  %d iteraciones.',...
 dataset_name,units_hidden_layer, length(error_vec));

%Plot circle
subplot(2,2,1);
t = linspace(0,2*pi,100)';
circsx = 7.*cos(t) + 10;
circsy = 7.*sin(t) + 10;
plot(circsx,circsy,'k');

%Plot points
for i = 1:length(training_set)
  hold on;
  [_, out] = forward_propagation(training_set(i,1:2), done_weights_1,...
   done_weights_2, units_hidden_layer);
  if (out  < 0)
    plot(training_set(i,1),training_set(i,2),'r*');
  else
    plot(training_set(i,1),training_set(i,2),'b*');
  endif
endfor

validate(done_weights_1, done_weights_2, units_hidden_layer);

%Plot error
subplot(2,2,[3,4]);
plot(error_vec,'LineWidth',2);
title(title_plot);