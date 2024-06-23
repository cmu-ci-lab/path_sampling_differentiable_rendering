### common test functions

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import mitsuba as mi
import drjit as dr

scenes = {
          'dragon_adaptive': 'scenes/dragon/scene_adaptive.xml',
          'dragon_not_adaptive': 'scenes/dragon/scene_not_adaptive.xml',
          'cbox_adaptive': 'scenes/cbox/scene_adaptive.xml',
          'cbox_not_adaptive': 'scenes/cbox/scene_not_adaptive.xml',
         }

colors = ['#b2df8a', '#33a02c', '#cab2d6', '#6a3d9a', '#a6cee3', '#1f78b4', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']

def print_params(params, keys):
    for key in keys:
        print(key + ":", params[key])

def print_info(arr, name=""):
    print(name, "min", np.min(arr), ", max", np.max(arr), ", sum", np.sum(arr), ", median", np.median(arr), ", mean", np.mean(arr))

def rel_sq_error(values, ref):
    epsilon = 1e-3
    return np.square((np.array(values) - ref) / (ref + epsilon))

def save_img_grad(img_grad, filename, vlim=5):
    img_grad = np.sum(img_grad, axis=-1)
    plt.clf(); plt.figure(figsize=(8,6))
    pc = plt.imshow(img_grad, cmap=cm.coolwarm, vmin=-vlim, vmax=vlim)
    plt.axis('off');
    # cb = plt.colorbar(pc, pad=0.05, ticks=[-vlim, 0, vlim], shrink=0.6)
    # cb.outline.set_visible(False); cb.ax.tick_params(size=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    print("Saving image gradient, vlim =", vlim)

def save_pixel_weights(weights, filename, vlim=0.001):
    plt.clf(); plt.figure(figsize=(8,6))
    cmap = sns.dark_palette("xkcd:golden", as_cmap=True)
    pc = plt.imshow(weights, cmap=cmap, vmin=0, vmax=vlim)
    plt.axis('off')
    # print("Saving pixel weights, vlim =", vlim)
    # cb = plt.colorbar(pc, pad=0.05, ticks=[0, vlim]) #, shrink=0.6)
    # cb.outline.set_visible(False); cb.ax.tick_params(size=0) #; cb.ax.locator_params(nbins=2)
    plt.savefig(filename+"weight_map.png", bbox_inches='tight', pad_inches=0)


def compute_variance_forward(scene_path, key, sensor, integrator, integrator_grad, spp, spp_grad, img_ref, out_file, vlim=40, new_param=None):
    var_iterations = 100; vlim = vlim
    height = img_ref.shape[0]; width = img_ref.shape[1]
    grad_imgs = np.zeros((var_iterations, height, width, 3))

    # load scene
    scene = mi.load_file(scene_path)
    params = mi.traverse(scene)
    if new_param:
        params[key] = new_param; params.update();

    for i in range(var_iterations):
        # load scene
        scene = mi.load_file(scene_path)
        params = mi.traverse(scene)
        if new_param:
            params[key] = new_param
        dr.enable_grad(params[key])
        params.update();

        # render and compute gradients
        img_ = mi.render(scene, params, sensor=sensor, integrator=integrator, integrator_grad=integrator_grad, spp=spp, spp_grad=spp_grad, seed=i)
        dr.forward(params[key]); dI_dtheta = dr.grad(img_) # image gradient
        grad_imgs[i] = np.array(dI_dtheta).reshape(height, width, 3)

    # compute variance of gradients
    variance_img = np.mean(np.var(grad_imgs, axis=0), axis=-1) # average variance across channels

    plt.clf(); pc = plt.imshow(variance_img, cmap=cm.jet, vmin=0, vmax=vlim)
    plt.axis('off');
    # cb = plt.colorbar(pc, pad=0.05, ticks=[0, vlim], shrink=0.6)
    # cb.outline.set_visible(False); cb.ax.tick_params(size=0)
    plt.savefig(out_file+'_variance.png', bbox_inches='tight', pad_inches=0)
    print_info(variance_img, "variance of image gradients:")
    print("saving variance image, vlim =", vlim)
    return variance_img


def compute_variance_reverse(scene_path, key, param_value, sensor, integrator, integrator_grad, spp, spp_grad, img_ref, adaptive_sampling=False):
    var_iterations = 100
    epsilon = 1e-3
    loss_grads = np.zeros(var_iterations)

    if adaptive_sampling:
        assert sensor.adaptive_sampling()

    print('computing variance:', spp, spp_grad, ', new param value:', param_value)
    for i in range(var_iterations):
        # load scene
        scene = mi.load_file(scene_path)
        params = mi.traverse(scene)
        params[key] = param_value # perturbed parameter
        dr.enable_grad(params[key])
        params.update();

        # render and compute gradients
        img_ = mi.render(scene, params, sensor=sensor, integrator=integrator, integrator_grad=integrator_grad, spp=spp, spp_grad=spp_grad, seed=i)
        masked_ref = img_ref
        loss = dr.mean(dr.sqr((img_ - masked_ref) / dr.maximum(dr.abs(masked_ref), epsilon))) # use relative loss
        dr.backward(loss)
        dL_dtheta = dr.grad(params[key])[0]
        loss_grads[i] = dL_dtheta

    # compute variance of gradients
    variance_loss_grads = np.var(loss_grads, axis=0)
    print("variance of loss gradients:", variance_loss_grads)
    return variance_loss_grads