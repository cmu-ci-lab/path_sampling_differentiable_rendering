### Test variance of image gradients with and without differential sampling
### no adaptive sampling
### change use_mis here and in prb_basic_diff and recompile

import drjit as dr
import mitsuba as mi

from common import scenes, print_params, save_img_grad, compute_variance_forward

mi.set_variant('cuda_ad_rgb')

max_depth = -1; rr_depth = 4294967295
bsdf_integrator = mi.load_dict({'type': 'prb_basic', 'max_depth': max_depth, 'rr_depth': rr_depth})
diff_integrator = mi.load_dict({'type': 'prb_basic_diff', 'max_depth': max_depth, 'rr_depth': rr_depth})

scene_key = 'dragon'
scene_name = scenes[f'{scene_key}_not_adaptive']
scene = mi.load_file(scene_name)
params = mi.traverse(scene); sensor = scene.sensors()[0]
key = 'diff_bsdf.weight.value'
dr.enable_grad(params[key])
params.update(); print_params(params, [key])


spp = 2; spp_ref = 512; vlim_var = 1; vlim_grad = 0.2
path = f'out/variance-img/{scene_key}-spp-{spp}'

# BSDF sampling --------------------
print('\nReference render')
print('adaptive_sampling:', sensor.adaptive_sampling())
img_ref = mi.render(scene, params, integrator=bsdf_integrator, seed=0, spp=spp_ref)
mi.util.convert_to_bitmap(img_ref, uint8_srgb=True).write(f"{path}/img_ref.png")

print("\nReference image gradient")
dr.forward(params[key]); dI_dtheta_ref = dr.grad(img_ref)
save_img_grad(dI_dtheta_ref, f"{path}/grad_ref.png", vlim=vlim_grad)

# Test variance --------------------
print('\nBRDF sampling')
compute_variance_forward(scene_name, key, sensor, bsdf_integrator, bsdf_integrator, spp, spp, img_ref, f'{path}/bsdf', vlim=vlim_var)

print("\nDifferential sampling with MIS")
compute_variance_forward(scene_name, key, sensor, bsdf_integrator, diff_integrator, spp, spp, img_ref, f'{path}/mis-diff', vlim=vlim_var)

use_nee = True
if use_nee:
    print("\nBRDF sampling + NEE")
    bsdf_integrator = mi.load_dict({'type': 'prb', 'max_depth': max_depth, 'rr_depth': rr_depth})
    compute_variance_forward(scene_name, key, sensor, bsdf_integrator, bsdf_integrator, spp, spp, img_ref, f'{path}/bsdf-nee', vlim=vlim_var)

    print("\nDifferential sampling with MIS + NEE")
    diff_integrator = mi.load_dict({'type': 'prb_diff', 'max_depth': max_depth, 'rr_depth': rr_depth})
    compute_variance_forward(scene_name, key, sensor, bsdf_integrator, diff_integrator, spp, spp, img_ref, f'{path}/mis-diff-nee', vlim=vlim_var)