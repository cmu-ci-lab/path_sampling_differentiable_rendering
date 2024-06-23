import matplotlib.pyplot as plt
import matplotlib.cm as cm

import drjit as dr
import mitsuba as mi
import numpy as np

from common import scenes, print_params, print_info, save_img_grad

mi.set_variant('cuda_ad_rgb')


spp = 2048
max_depth = -1; rr_depth = 4294967295
scene_key = 'dragon'
scene_name = scenes[f'{scene_key}_not_adaptive']
scene = mi.load_file(scene_name)
vlim = 0.3

params = mi.traverse(scene)
key = 'diff_bsdf.weight.value'

# optional parameter update
params[key] = 0.4

dr.enable_grad(params[key])
params.update(); print_params(params, [key])

# BSDF sampling --------------------
print('\nReference render')
bsdf_integrator = mi.load_dict({'type': 'prb', 'max_depth': max_depth, 'rr_depth': rr_depth})
print('adaptive_sampling:', scene.sensors()[0].adaptive_sampling())
img_ref = mi.render(scene, params, integrator=bsdf_integrator, seed=0, spp=spp)
mi.util.convert_to_bitmap(img_ref, uint8_srgb=True).write("out/img_ref.png")

print("\nReference image gradient")
dr.forward(params[key]); dI_dtheta_ref = dr.grad(img_ref)
save_img_grad(dI_dtheta_ref, "out/grad_ref.png", vlim=vlim)
print_info(dI_dtheta_ref)

# Differential sampling --------------------
print('\nDifferential sampling render')
diff_integrator = mi.load_dict({'type': 'prb_diff', 'max_depth': max_depth, 'rr_depth': rr_depth})
print('adaptive_sampling:', scene.sensors()[0].adaptive_sampling())
img = mi.render(scene, params, integrator=diff_integrator, seed=0, spp=spp)
mi.util.convert_to_bitmap(img, uint8_srgb=True).write("out/img_diff.png")

print("\nDifferential sampling image gradient")
dr.forward(params[key]); dI_dtheta = dr.grad(img)
save_img_grad(dI_dtheta, "out/grad_diff.png", vlim=vlim)
print_info(dI_dtheta)
