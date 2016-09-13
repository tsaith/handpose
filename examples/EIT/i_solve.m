
    % Inverse Model: R= inv(J'*J)*J'*V
    %    This corresponds to the least squares solution
    function img= i_solve( i_mdl, data )
      % Normally the Jacobian depends on an image. Create a dummy one here
      i_img= eidors_obj('image','Unused');
      f_mdl= i_mdl.fwd_model;
      J = calc_jacobian( f_mdl, i_img);

      img.name= 'solved by i_solve';
      img.elem_data= (J'*J)\J'* data.meas;
      img.inv_model= i_mdl;
        data.meas= V;

