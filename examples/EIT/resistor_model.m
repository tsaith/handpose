    %
    % create FEM model structure
    %
    % Fwd model:
    %  Two nodes are in space at [1,1,1] and [2,2,2]
    %  The resistor is connected between them

    r_mdl.name = 'demo resistor model';
    r_mdl.nodes= [1,1,1;  2,2,2];
    r_mdl.elems= [1;2];
    r_mdl.solve=      @f_solve;
    r_mdl.jacobian=   @c_jacobian;

    %
    % create FEM model electrode definitions
    %

    r_mdl.electrode(1).z_contact= 10; % ohms
    r_mdl.electrode(1).nodes=     1;
    r_mdl.gnd_node= 2;

    %
    % create stimulation and measurement patterns
    % patterns are 0.010,0.020,0.030 mA

    for i=1:3
        r_mdl.stimulation(i).stimulation= 'mA';
        r_mdl.stimulation(i).stim_pattern= ( 0.010*i );
        r_mdl.stimulation(i).meas_pattern= 1; % measure electrode 1
    end

    r_mdl= eidors_obj('fwd_model', r_mdl);

    %
    % simulate data for medium with R=1 kohms
    % This medium is called an 'image'
    %

    img_1k = eidors_obj('image', 'homogeneous image', ...
                         'elem_data', 1e3, ...
                         'fwd_model', r_mdl );

    data_1k =fwd_solve( r_mdl, img_1k );

    %
    % add noise to simulated data
    %

    data_noise= eidors_obj('data', 'noisy data', ...
                           'meas', data_1k.meas + 1e-3*randn(3,1));

    %
    % create inverse model
    %

    % create an inv_model structure of name 'demo_inv'
    r_inv.name=  'Resistor Model inverse';
    r_inv.solve= @i_solve;
    r_inv.reconst_type= 'static';
    r_inv.fwd_model= r_mdl;
    r_inv= eidors_obj('inv_model', r_inv);

    %
    % solve inverse model');
    %

    R= inv_solve( r_inv, data_1k );
    fprintf('R calculated with clean data= %5.3f kOhm\n', R.elem_data / 1000 );

    R= inv_solve( r_inv, data_noise );
    fprintf('R calculated with noisy data= %5.3f kOhm\n', R.elem_data / 1000 );


