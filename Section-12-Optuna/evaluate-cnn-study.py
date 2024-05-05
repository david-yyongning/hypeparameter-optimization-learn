import optuna
# pip install potly if you don't have it

# Load the search results of study
study_name = 'cnn_study'
storage_name = 'sqlite:///{}.db'.format(study_name)
study = optuna.load_study(
    study_name=study_name,
    storage=storage_name,
)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")

# Optimization history
fig = optuna.visualization.plot_optimization_history(study).show()

# Contour plot - 2D plots
fig = optuna.visualization.plot_contour(
    study, 
    params=['num_conv_layers', 'num_dense_layers', 'optimizer', 'units'],
    ).show()

# Plot slice
fig = optuna.visualization.plot_slice(
    study,
    params=['num_conv_layers', 'num_dense_layers', 'optimizer', 'units'],
    ).show()

# Parameter importance
fig = optuna.visualization.plot_param_importances(study).show()

# Plot parallel coordinate
fig = optuna.visualization.plot_parallel_coordinate(
    study,
    params=['num_conv_layers', 'num_dense_layers', 'optimizer', 'units'],
    ).show()

# Compare 2 or more studies
optuna.visualization.plot_edf([study]).show()

study_name2 = 'cnn_study_2'
storage_name2 = 'sqlite:///{}.db'.format(study_name2)
study2 = optuna.load_study(
    study_name=study_name2,
    storage=storage_name2,
)
optuna.visualization.plot_edf([study, study2]).show()