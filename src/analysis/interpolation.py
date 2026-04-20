from plot_utils import plot_interpolation_curve(...)
from utils import compute_fisher_vectors(...)

def interpolate():
    # Has to be SO(n) manifold interpolation
    if start_model == None:
        # Assume all weights are Identity, which corresponds to pretrained.
    ...


def make_loss():
    ...

def interpolate_model(start_model, end_model, interpolation_grid, ...):
    for alpha in interpolation_grid:
        interpolated_weights = interpolate(start_model, end_model, alpha=alpha)
        interpolation_losses.append(llm_loss(interpolated_weights))

    return interpolation_losses

def main():
    start_model  # can be a fine-tuned model or pretrained model (None)

    for model in all_models:
        loss = make_loss(model)
        interpolation_losses = interpolate_model(...)
        plot_interpolation_curve(...)