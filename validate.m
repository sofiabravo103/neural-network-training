function [perc_aciertos] = validate (weights_1, weights_2, units_hidden_layer)
  pts = [];
  L = 0:0.2:20;
  p = 1;

  for i = L
    for j = L
      pts(p,1) = i;
      pts(p,2) = j;
      p += 1;
    endfor
  endfor

  results = [];
  targets = [];
  for i = 1:length(pts)
    [_, out] = forward_propagation(pts(i,:), weights_1,...
      weights_2, units_hidden_layer);
    results(i) = out;

    value = ((pts(i,1) - 10) ^ 2 ) + ((pts(i,2) - 10) ^ 2);
    if (value <= 49)
      targets(i) = -1;
    else
      targets(i) = 1;
    endif
  endfor


  subplot(2,2,2);
  t = linspace(0,2*pi,100)';
  circsx = 7.*cos(t) + 10;
  circsy = 7.*sin(t) + 10;
  plot(circsx,circsy,'k');

  aciertos = 0;
  for i = 1:length(pts)
    hold on;
    if (results(i) < 0)
      if (targets(i) < 0)
        plot(pts(i,1),pts(i,2),'r*');
        aciertos += 1;
      else
        plot(pts(i,1),pts(i,2),'k*');
      endif
    else
      if (targets(i) > 0)
        plot(pts(i,1),pts(i,2),'b*');
        aciertos += 1;
      else
        plot(pts(i,1),pts(i,2),'k*');
      endif
    endif
  endfor

  perc_aciertos = (aciertos * 100) / length(pts);
  title(sprintf('Porcentaje de aciertos: %d%s', perc_aciertos,'%'));

endfunction