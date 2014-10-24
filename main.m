sizes = [500; 1000; 2000];
units_hidden_layer = 5:10;
figure_no = 1;

for s = 1:3
  dataset_name = sprintf('Datos del proyecto con %d puntos.',sizes(s));
  for units = units_hidden_layer
    ts = dlmread(sprintf('datos%d_1', sizes(s)));
    train_and_validate(ts,units,dataset_name, figure_no);
    figure_no += 1;
  endfor
endfor

for  s = 1:3
  dataset_name = sprintf('Generado por el grupo con %d puntos.',sizes(s));
  for units = units_hidden_layer
    ts = dlmread(sprintf('datos%d_2', sizes(s)));
    train_and_validate(ts,units,dataset_name, figure_no);
    figure_no += 1;
  endfor
endfor
