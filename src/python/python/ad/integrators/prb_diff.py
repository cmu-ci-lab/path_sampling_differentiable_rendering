from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from .common import RBIntegrator

class PRBDiffIntegrator(RBIntegrator):
    r"""
    .. _integrator-prb_diff:

    Path Replay Backpropagation with Differential + Adaptive Sampling (:monosp:`prb_diff`)
    -------------------------------------------

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: 6)

     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (Default: 5)

    This plugin implements a basic Path Replay Backpropagation (PRB) integrator
    with the following properties:

    - Emitter sampling (a.k.a. next event estimation).

    - Russian Roulette stopping criterion.

    - No reparameterization. This means that the integrator cannot be used for
      shape optimization (it will return incorrect/biased gradients for
      geometric parameters like vertex positions.)

    - Detached sampling. This means that the properties of ideal specular
      objects (e.g., the IOR of a glass vase) cannot be optimized.

    See ``prb_basic.py`` for an even more reduced implementation that removes
    the first two features.

    See the papers :cite:`Vicini2021` and :cite:`Zeltner2021MonteCarlo`
    for details on PRB, attached/detached sampling, and reparameterizations.

    .. tabs::

        .. code-tab:: python

            'type': 'prb',
            'max_depth': 8
    """

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               p_sample_diff: float = 0,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum,
               mi.Bool, mi.Spectrum]:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # Differential BSDF evaluation contexts
        ctx_diff_pos = mi.BSDFContext()
        ctx_diff_pos.type_mask |= mi.BSDFFlags.DifferentialSamplingPositive
        ctx_diff_neg = mi.BSDFContext()
        ctx_diff_neg.type_mask |= mi.BSDFFlags.DifferentialSamplingNegative

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)                            # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes
        sampled_diff_vertex = mi.Bool(False)          # Has the differential vertex already been sampled?
        Lr_dir_sum = mi.Spectrum(0)                   # Direct lighting accumulator
        pdf_path = mi.Float(0)
        w = mi.Float(1)

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        prev_pdf_path   = mi.Float(0)
        prev_w          = mi.Float(1)

        # Record the following loop in its entirety
        loop = mi.Loop(name="Path Replay Backpropagation (%s)" % mode.name,
                       state=lambda: (sampler, ray, depth, L, δL, β, η, active,
                                      sampled_diff_vertex, Lr_dir_sum, pdf_path, w,
                                      prev_si, prev_bsdf_pdf, prev_bsdf_delta, prev_pdf_path, prev_w))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            active_next = mi.Bool(active)

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All,
                                         coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            prev_em_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            prev_em_valid = prev_em_pdf > 0

            # Weight both direct and indirect lighting by 0.5 (no MIS)
            # mis = dr.select(prev_em_valid, 0.5, 1)
            # with dr.resume_grad(when=not primal):
            #     Le = β * ds.emitter.eval(si) * mis / (pdf_path + w)

            # Use MIS weight for direct/indirect lighting
            mis_pdf_path = dr.select(prev_em_valid, (prev_pdf_path + prev_w) * prev_em_pdf / prev_bsdf_pdf, 0)
            with dr.resume_grad(when=not primal):
                Le = β * ds.emitter.eval(si) / (pdf_path + w + mis_pdf_path)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            with dr.resume_grad(when=not primal):
                if not primal:
                    # Given the detached emitter sample, *recompute* its
                    # contribution with AD to enable light source optimization
                    ds.d = dr.replace_grad(ds.d, dr.normalize(ds.p - si.p))
                    em_val = scene.eval_emitter_direction(si, ds, active_em)
                    em_weight = dr.replace_grad(em_weight, dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0))
                    dr.disable_grad(ds.d)

                # Evaluate BSDF * cos(theta) differentiably
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)

                # Weight both direct and indirect lighting by 0.5 (no MIS)
                # mis_em = 0.5
                # Lr_dir = β * mis_em * bsdf_value_em * em_weight / (pdf_path + w)

            # Use MIS weight for direct/indirect lighting
            pdf_diff_em = bsdf.pdf(ctx_diff_pos, si, wo, mi.Bool(active_em))
            pdf_diff_em = dr.select(dr.isnan(pdf_diff_em), 0, pdf_diff_em)
            mis_pdf_em = (pdf_path + p_sample_diff * w * pdf_diff_em / (bsdf_pdf_em + 1e-10) + w * (1 - p_sample_diff)) * bsdf_pdf_em / (ds.pdf + 1e-10)
            with dr.resume_grad(when=not primal):
                Lr_dir = β * bsdf_value_em * em_weight / (pdf_path + w + mis_pdf_em)

            # ------------------ Detached BSDF sampling -------------------

            # Sample BSDF
            active_bsdf = mi.Bool(active_next)
            bsdf_sample, _ = bsdf.sample(bsdf_ctx, si,
                                         sampler.next_1d(),
                                         sampler.next_2d(),
                                         active_bsdf)

            # Sample positive component of BSDF derivative
            active_pos = mi.Bool(active_next)
            diff_pos_sample, _ = bsdf.sample(ctx_diff_pos, si,
                                             sampler.next_1d(),
                                             sampler.next_2d(),
                                             active_pos)

            # Sample negative component of BSDF derivative
            active_neg = mi.Bool(active_next)
            diff_neg_sample, _ = bsdf.sample(ctx_diff_neg, si,
                                             sampler.next_1d(),
                                             sampler.next_2d(),
                                             active_neg)

            # Determine if sampling from the BSDF or the BSDF derivative
            diff_valid = bsdf.use_differential_sampling()
            diff_vertex_prob = dr.select(sampled_diff_vertex | ~diff_valid, 0, p_sample_diff)
            sample_diff_bsdf = sampler.next_1d() < diff_vertex_prob # make this vertex differential
            sample_pos = sampler.next_1d() < 0.5 # pos/neg component of BSDF derivative

            active_next = dr.select(sample_diff_bsdf,
                                    dr.select(sample_pos, active_pos, active_neg),
                                    active_bsdf)
            bsdf_sample.wo = dr.select(sample_diff_bsdf,
                                       dr.select(sample_pos, diff_pos_sample.wo, diff_neg_sample.wo),
                                       bsdf_sample.wo)
            bsdf_sample.eta = dr.select(sample_diff_bsdf,
                                        dr.select(sample_pos, diff_pos_sample.eta, diff_neg_sample.eta),
                                        bsdf_sample.eta)
            bsdf_sample.sampled_type = dr.select(sample_diff_bsdf,
                                                 dr.select(sample_pos, diff_pos_sample.sampled_type, diff_neg_sample.sampled_type),
                                                 bsdf_sample.sampled_type)

            # Compute pdf values for the sampled direction
            pdf_bsdf = bsdf.pdf(bsdf_ctx, si, bsdf_sample.wo, mi.Bool(active_next))
            pdf_diff = bsdf.pdf(ctx_diff_pos, si, bsdf_sample.wo, mi.Bool(active_next))
            bsdf_sample.pdf = pdf_bsdf # use pdf of forward bsdf

            # Use pdf without path mixture pdf instead
            # pdf_total = p_sample_diff * pdf_diff + (1 - p_sample_diff) * pdf_bsdf
            # bsdf_sample.pdf = dr.select(sampled_diff_vertex, # if haven't sampled diff vertex, pdf = 0.5 * (diff_pdf + bsdf_sample.pdf)
            #                             bsdf_sample.pdf,
            #                             pdf_total)

            # Compute bsdf/pdf
            bsdf_weight = dr.select(bsdf_sample.pdf > 0,
                                    bsdf.eval(bsdf_ctx, si, bsdf_sample.wo, active_next) / bsdf_sample.pdf, 0)

            # ---- Update loop variables based on current interaction -----

            L = (L + Le) if primal else (L - Le - Lr_dir)
            Lr_dir_sum += Lr_dir
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            prev_pdf_path = mi.Float(pdf_path)
            prev_w = mi.Float(w)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # # Russian roulette stopping probability (must cancel out ior^2
            # # to obtain unitless throughput, enforces a minimum probability)
            # rr_prob = dr.minimum(β_max * η**2, .95)

            # # Apply only further along the path since, this introduces variance
            # rr_active = depth >= self.rr_depth
            # β[rr_active] *= dr.rcp(rr_prob)
            # rr_continue = sampler.next_1d() < rr_prob
            # active_next &= ~rr_active | rr_continue

            # ---------------- Update the path mixture pdf ----------------

            # Only update pdf value if differential sampling could have been used
            pdf_path += dr.select(active_next & diff_valid, p_sample_diff * w * dr.select(dr.isnan(pdf_diff), 0, pdf_diff) / (pdf_bsdf + 1e-10), 0)
            w *= dr.select(active_next & diff_valid, 1 - p_sample_diff, 1)

            # Update whether a differential vertex was sampled
            sample_diff_bsdf &= active_next # current vertex is not differential if there is no next vertex
            sampled_diff_vertex |= sample_diff_bsdf

            # ------------------ Differential phase only ------------------

            if not primal:
                with dr.resume_grad():
                    # 'L' stores the indirectly reflected radiance at the
                    # current vertex but does not track parameter derivatives.
                    # The following addresses this by canceling the detached
                    # BSDF value and replacing it with an equivalent term that
                    # has derivative tracking enabled. (nit picking: the
                    # direct/indirect terminology isn't 100% accurate here,
                    # since there may be a direct component that is weighted
                    # via multiple importance sampling)

                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si.to_local(ray.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(dr.neq(bsdf_val_det, 0),
                                                 dr.rcp(bsdf_val_det), 0)

                    # Differentiable version of the reflected indirect
                    # radiance. Minor optional tweak: indicate that the primal
                    # value of the second term is always 1.
                    Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_det * bsdf_val)

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_dir + Lr_ind

                    if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            depth[si.is_valid()] += 1
            active = active_next

        # ------------- Include direct lighting contributions -------------
        if primal:
            L += Lr_dir_sum

        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L                    # State for the differential phase
        )

mi.register_integrator("prb_diff", lambda props: PRBDiffIntegrator(props))