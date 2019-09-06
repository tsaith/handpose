    % Jacobian:      J= dV/dR =I = [I1; I2; I3]
    function J= c_jacobian( f_mdl, img)
      n_stim= length( f_mdl.stimulation );
      J= zeros(n_stim, 1);
      for i=1:n_stim
        J(i)     = f_mdl.stimulation(i).stim_pattern / 1000; % mA
      end
    
