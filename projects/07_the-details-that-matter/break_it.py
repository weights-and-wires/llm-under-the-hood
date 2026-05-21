"""
Project 7: BREAK IT experiment.

Deliberately sabotages one mechanism from build.py to show what happens
when it's removed. Compare outputs to the working version.
"""

self.norm1 = make_norm(config.norm_type, config.d_model)
self.norm2 = make_norm(config.norm_type, config.d_model)

def norm_for_layer(layer_idx, n_layers, d_model):
    if layer_idx < n_layers // 2:
        return LayerNorm(d_model)
    return RMSNorm(d_model)

self.norm1 = norm_for_layer(layer_idx, config.n_layers, config.d_model)
self.norm2 = norm_for_layer(layer_idx, config.n_layers, config.d_model)
