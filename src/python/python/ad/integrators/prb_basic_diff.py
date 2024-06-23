from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from .common import RBIntegrator

class BasicPRBDiffIntegrator(RBIntegrator):
    r"""
    .. _integrator-prb_basic_diff:

    Basic Path Replay Backpropagation with Differential + Adaptive Sampling (:monosp:`prb_basic_diff`)
    ---------------------------------------------------------

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: 6)

    Basic Path Replay Backpropagation-style integrator *without* next event
    estimation, multiple importance sampling, Russian Roulette, and
    reparameterization. The lack of all of these features means that gradients
    are noisy and don't correctly account for visibility discontinuities. The
    lack of a Russian Roulette stopping criterion means that generated light
    paths may be unnecessarily long and costly to generate.

    This class is not meant to be used in practice, but merely exists to
    illustrate how a very basic rendering algorithm can be implemented in
    Python along with efficient forward/reverse-mode derivatives. See the file
    ``prb.py`` for a more feature-complete Path Replay Backpropagation
    integrator, and ``prb_reparam.py`` for one that also handles visibility.

    .. tabs::

        .. code-tab:: python

            'type': 'prb_basic',
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
        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)                            # Path throughput weight
        active = mi.Bool(active)                      # Active SIMD lanes
        sampled_diff_vertex = mi.Bool(False)          # Has the differential vertex already been sampled?
        pdf_path = mi.Float(0)
        w = mi.Float(1)

        # Record the following loop in its entirety
        loop = mi.Loop(name="Path Replay Backpropagation (%s)" % mode.name,
                       state=lambda: (sampler, ray, depth, L, δL, β, active,
                                      sampled_diff_vertex, pdf_path, w))

        while loop(active):
            # ---------------------- Direct emission ----------------------

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.

            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray)

                # Differentiable evaluation of intersected emitter / envmap
                Le = β * si.emitter(scene).eval(si) / (pdf_path + w) # divide by path mixture pdf

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # Get the BSDF. Potentially computes texture-space differentials.
            bsdf = si.bsdf(ray)

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

            sample_diff_bsdf &= active_next # vertex is not differential if there is no next vertex

            # ---- Update loop variables based on current interaction -----

            L = L + Le if primal else L - Le
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            β *= bsdf_weight

            # Don't run another iteration if the throughput has reached zero
            active_next &= dr.any(dr.neq(β, 0))
            sampled_diff_vertex |= (sample_diff_bsdf & active_next)

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
                    # 'L' stores the reflected radiance at the current vertex
                    # but does not track parameter derivatives. The following
                    # addresses this by canceling the detached BSDF value and
                    # replacing it with an equivalent term that has derivative
                    # tracking enabled.

                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si.to_local(ray.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_detach = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_detach = dr.select(dr.neq(bsdf_val_detach, 0),
                                                    dr.rcp(bsdf_val_detach), 0)

                    # Differentiable version of the reflected radiance. Minor
                    # optional tweak: indicate that the primal value of the
                    # second term is 1.
                    Lr = L * dr.replace_grad(1, inv_bsdf_val_detach * bsdf_val)

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            depth[si.is_valid()] += 1
            active = active_next

        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L                    # State the for differential phase
        )

mi.register_integrator("prb_basic_diff", lambda props: BasicPRBDiffIntegrator(props))
