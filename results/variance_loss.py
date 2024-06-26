### Test variance of loss gradients with and without adaptive sampling
### with NEE

import numpy as np
import drjit as dr
import mitsuba as mi

from common import scenes, print_params, save_pixel_weights, compute_variance_reverse

mi.set_variant('cuda_ad_rgb')

max_depth = -1; rr_depth = 4294967295; epsilon = 1e-3
bsdf_integrator = mi.load_dict({'type': 'prb', 'max_depth': max_depth, 'rr_depth': rr_depth})
diff_integrator = mi.load_dict({'type': 'prb_diff', 'max_depth': max_depth, 'rr_depth': rr_depth})


spp = 3; spp_grad = 1; spp_ref = 1024
param_new = 0.9
key = 'diff_bsdf.weight.value'
scene_key = 'dragon'
path = f'out/variance-loss/{scene_key}-spp-{spp}-{spp_grad}'

scene_name = scenes[f'{scene_key}_not_adaptive']
scene = mi.load_file(scene_name)
params = mi.traverse(scene); sensor = scene.sensors()[0]
dr.enable_grad(params[key]); params.update()
print_params(params, [key])

# BSDF sampling --------------------
print('\nReference render')
print('adaptive_sampling:', sensor.adaptive_sampling())
img_ref = mi.render(scene, params, integrator=bsdf_integrator, seed=0, spp=spp_ref)
mi.util.convert_to_bitmap(img_ref, uint8_srgb=True).write(f"{path}/img_ref.png")

print('\n--- Without adaptive sampling')
compute_variance_reverse(scene_name, key, param_new, sensor, bsdf_integrator, bsdf_integrator, spp, spp_grad, img_ref, adaptive_sampling=False)

# Adaptive sampling --------------------
print('\n--- Computing pixel weights')
scene_name = scenes[f'{scene_key}_adaptive']
scene = mi.load_file(scene_name)
params = mi.traverse(scene); sensor = scene.sensors()[0]
params[key] = param_new; dr.enable_grad(params[key]); params.update()
print_params(params, [key])

img_weights = mi.render(scene, params, sensor=sensor, integrator=bsdf_integrator, spp=spp, seed=2000)
pixel_weights = np.sum(dr.abs(img_weights - img_ref) / np.power(dr.maximum(dr.abs(img_ref), epsilon), 2), axis=-1) # relative_loss
pixel_weights = pixel_weights / np.sum(pixel_weights) # normalize
sensor.set_pixel_weights(pixel_weights.flatten()) # set weights
height = img_ref.shape[0]; width = img_ref.shape[1]
save_pixel_weights(np.array(sensor.pixel_weights()).reshape(height, width), f'{path}/', vlim=4/(height*width))

print("\n--- Adaptive sampling")
compute_variance_reverse(scene_name, key, param_new, sensor, bsdf_integrator, bsdf_integrator, spp, spp_grad, img_ref, adaptive_sampling=True)

use_diff_sampling = True
if use_diff_sampling:
    print("\n--- Adaptive + differential sampling")
    compute_variance_reverse(scene_name, key, param_new, sensor, bsdf_integrator, diff_integrator, spp, spp_grad, img_ref, adaptive_sampling=True)