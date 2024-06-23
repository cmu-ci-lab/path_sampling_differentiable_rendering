import matplotlib.pyplot as plt
import seaborn as sns

import drjit as dr
import mitsuba as mi
import numpy as np
from common import scenes, colors, save_pixel_weights, rel_sq_error
sns.set_palette(colors)

mi.set_variant('cuda_ad_rgb')

### Parameters

max_depth = -1; rr_depth = 4294967295
ref_spp = 1024; epsilon = 1e-3
save_weights = True

lr = 0.002
spp = 6; spp_grad = 2; spp_label = f'{spp} primal, {spp_grad} adjoint'
num_iterations = 500
height = 256; width = 256
scene_name = "Sphere"; scene_key = 'cbox'
scene_path = scenes[f'{scene_key}_adaptive']; scene_path_not_adaptive = scenes[f'{scene_key}_not_adaptive']
key = 'diff_bsdf.alpha.value'

alpha_ref = 0.3; alpha_init = 0.01
out_path = "out/inverse-rendering/sphere/"

### Inverse rendering

def run_optimization(differential_sampling, adaptive_sampling):
    print("\nDifferential sampling:", differential_sampling)
    print("Adaptive sampling:", adaptive_sampling)
    method = "_diff" if differential_sampling else "_bsdf"
    method += "_adaptive" if adaptive_sampling else ""

    # Render reference image (BSDF sampling, 1024 spp)
    scene_ref = mi.load_file(scene_path)
    params_ref = mi.traverse(scene_ref)
    params_ref[key] = alpha_ref; params_ref.update();
    integrator_bsdf = mi.load_dict({'type': 'prb', 'max_depth': max_depth, 'rr_depth': rr_depth})
    img_ref = dr.detach(mi.render(scene_ref, params_ref, integrator=integrator_bsdf, spp=ref_spp, seed=10000))
    mi.util.write_bitmap(out_path+"reference_img_"+str(alpha_ref)+".png", img_ref) # reference image
    print("Reference alpha:", alpha_ref)

    # Render initial image (BSDF sampling, 1024 spp)
    scene = mi.load_file(scene_path) if adaptive_sampling else mi.load_file(scene_path_not_adaptive)
    params = mi.traverse(scene); sensor = scene.sensors()[0]
    params[key] = alpha_init; params.update();
    img = mi.render(scene, params, integrator=integrator_bsdf, spp=ref_spp, seed=20000)
    mi.util.write_bitmap(out_path+"initial_img_"+str(alpha_init)+".png", img) # initial image
    print("Initial alpha:", alpha_init)

    # Set up optimization
    if differential_sampling:
        integrator_grad = mi.load_dict({'type': 'prb_diff', 'max_depth': max_depth, 'rr_depth': rr_depth})
    else:
        integrator_grad = integrator_bsdf
    opt = mi.ad.Adam(lr=lr)
    opt[key] = params[key]
    params.update(opt);

    # Set sensor's per-pixel weights
    if adaptive_sampling:
        img = mi.render(scene, params, integrator=integrator_bsdf, spp=spp, seed=0)
        diff = img - img_ref
        loss = dr.mean(dr.sqr(diff / (img_ref + epsilon)))
        pixel_weights = np.sum(dr.abs(diff) / np.power(img_ref + epsilon, 2), axis=-1)
        pixel_weights = pixel_weights / np.sum(pixel_weights) # normalize
        sensor.set_pixel_weights(pixel_weights.flatten()) # set weights

    # Optimization
    losses = []; alphas = [params[key][0]]
    for it in range(num_iterations):        
        # Perform the differentiable light transport simulation
        img = mi.render(scene, params, sensor=sensor, integrator=integrator_bsdf, integrator_grad=integrator_grad, spp=spp, spp_grad=spp_grad, seed=it)

        # Compute error
        diff = img - img_ref
        loss = dr.mean(dr.sqr(diff / (img_ref + epsilon)))

        # Update sensor's per-pixel weights
        if adaptive_sampling:
            pixel_weights = np.sum(dr.abs(img - img_ref) / np.power(img_ref + epsilon, 2), axis=-1)
            pixel_weights = pixel_weights / np.sum(pixel_weights) # normalize
            sensor.set_pixel_weights(pixel_weights.flatten()) # set weights
            if it%100 == 0 and save_weights:
                save_pixel_weights(np.array(sensor.pixel_weights()).reshape(width, height), f'{out_path}weights/it{it}-{spp}-{spp_grad}-', vlim=4/(height*width))
        
        dr.backward(loss) # backpropagate gradients
        opt.step() # gradient step
        opt[key] = dr.clamp(opt[key], 0.0, 1.0) # clamp to reasonable range
        params.update(opt); # propagate changes to the scene

        losses.append(loss[0])
        alphas.append(params[key][0])
        print(f"Iteration {it:02d}: loss={loss[0]:4f}, alpha={params[key][0]:4f}", end='\r')

    alpha_final = params[key][0]
    img_final = mi.render(scene, params, integrator=integrator_bsdf, spp=ref_spp, seed=1000)
    mi.util.write_bitmap(out_path+f"final_img{method}.png", img_final)
    print("Final alpha:", alpha_final)

    # Show optimization losses
    print(f"Iteration {it:02d}: loss={loss[0]:4f}, alpha={params[key][0]:4f}", end='\n')
    return losses, alphas


### Run optimizations

losses_ff, alphas_ff = run_optimization(False, False) # BRDF sampling
losses_ft, alphas_ft = run_optimization(False, True) # BRDF + adaptive sampling
losses_tf, alphas_tf = run_optimization(True, False) # Differential sampling
losses_tt, alphas_tt = run_optimization(True, True) # Differential + adaptive sampling
plt.figure(figsize=(8,6))

plt.clf()
plt.plot(losses_ff, label='BRDF sampling')
plt.plot(losses_ft, label='BRDF + adaptive sampling')
plt.plot(losses_tf, label='Differential sampling')
plt.plot(losses_tt, label='Differential + adaptive sampling')
plt.xlabel('Iteration'); plt.ylabel('Relative MSE of image')
plt.legend()
plt.savefig(out_path+'opt_mse_relative.png')

plt.clf()
alphas_ref = np.repeat(alpha_ref, num_iterations+1)
plt.plot(rel_sq_error(alphas_ff, alphas_ref), label='BRDF sampling')
plt.plot(rel_sq_error(alphas_ft, alphas_ref), label='BRDF + adaptive sampling')
plt.plot(rel_sq_error(alphas_tf, alphas_ref), label='Differential sampling')
plt.plot(rel_sq_error(alphas_tt, alphas_ref), label='Differential + adaptive sampling')
plt.xlabel('Iteration'); plt.ylabel('Relative square error of parameter')
plt.legend()
plt.savefig(out_path+'opt_mse_params_relative.png')