    % Forward Model:
    % For each stimulation there is I1 into Node1
    %  Node2 is connected to gnd with Zcontact
    %
    % each stim has one measurement pattern where
    %  Vmeas= Meas_pat * Node1
    %       = Meas_pat * I1 * ( Zcontact*2 + R )
    %
    % Thus
    %  V= IR    => [V1;V2;V3] = [I1;I2*I3]*(R + 2*Zcontact)
    function data =f_solve( f_mdl, img )
      R= img.elem_data;

      n_stim= length( f_mdl.stimulation );
      V= zeros(n_stim, 1);

      for i=1:n_stim
        I        = f_mdl.stimulation(i).stim_pattern / 1000; % mA
        meas_pat = f_mdl.stimulation(i).meas_pattern;

        stim_elec= find( I );
        zc       = f_mdl.electrode( stim_elec ).z_contact;

        V(i)= meas_pat * I * ( R + 2*zc);
      end

      data.name='resistor model data';

