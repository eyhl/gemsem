# gemsem

Introduction from documentation:
In the following the key functionality and features in the python module gemsem will be presented.
The module is meant for geomagnetic single epoch modelling, and implements many basic functions needed in this regard. 
It is specially focused on the core field evaluated at the core mantle boundary,
but the main modelling funcitons should be able to handle other design matrixes as well. 
The module serves can serve as a quick way to do a preliminary study of the data at hand, or serve as a basis for 
comparison with other types of modelling
The main structure of the module is seperated into 4 files:

- load.py−contains loading functions: data_loader()
- plot.py−contains plotting functions:power_spectrum(), plot_L_curve(), plot_hist_residuals(),plot_field_map()
- model.py−contains the three core moddelling functions:l2_model(), l2_model(), max_ent_model()
- utils.py−contains general utilities: L_curve(), design_SHA(), spherical_grid(), including backend functions:
  - _get_Pnm(), _define_icosahedral_grid(), _refine_triangle(), _project_grid_to_sphere(), _get_indices(), 
    _barycentric_coords(), _scalar_product(), _slerp(), _map_gridpoint_to_sphere(), _to_mjd2000(), 
    _revolutions_to_radians(), _sun_mjd2000()

In future versions the gemsem module will include a class which makes it easier to use the module. 
Another feature I would like to add is implementing the _get_Pnm() and design_SHA() functions in 
cython to improve speed, since these are core functions. In general many thing should be tested further and could
be finetuned.
