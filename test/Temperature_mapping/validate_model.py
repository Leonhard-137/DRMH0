"""Independent numerical checks of the temperature inversion; no campaign fit.

Only outputs/validation.json is generated.  All test problems are in memory.
Run with the existing astro environment: python -B validate_model.py.
"""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from numpy.polynomial.legendre import leggauss
from astropy import units as units
from astropy.modeling.models import BlackBody
from threadpoolctl import threadpool_limits

import temperature_mapping as model


ROOT = Path(__file__).resolve().parent


def check_temporal():
    """Compare analytic arcsine integration against uniform ring azimuths."""
    tp = np.linspace(0.0, 10.0, 21)
    td = np.array([0.0, 0.03, 0.4, 1.7, 4.7, 9.9, 10.0])
    delta = tp[1] - tp[0]
    nphi = 120000
    phi = (np.arange(nphi) + 0.5) * np.pi / nphi
    records = []
    for t0 in [0.001, 0.2, 1.0, 6.0]:
        analytic = model.temporal_weights(td, tp, t0, normalize=False)
        numeric = []
        for t in td:
            emitted = t + t0 * np.cos(phi)
            # Piecewise-linear interpolation of each independent unit impulse.
            numeric.append([
                np.interp(emitted, tp, impulse, left=0.0, right=0.0).mean()
                for impulse in np.eye(len(tp))
            ])
        # np.interp truncates at tp endpoints, whereas ordinary terminal hats
        # extend one bin beyond them. Integrate those two tails explicitly.
        numeric = np.asarray(numeric)
        for j, t in enumerate(td):
            emitted = t + t0 * np.cos(phi)
            left = (emitted < tp[0]) & (emitted >= tp[0] - delta)
            right = (emitted > tp[-1]) & (emitted <= tp[-1] + delta)
            numeric[j, 0] += np.sum((emitted[left] - tp[0] + delta) / delta) / nphi
            numeric[j, -1] += np.sum((tp[-1] + delta - emitted[right]) / delta) / nphi
        error = float(np.max(np.abs(analytic - numeric)))
        normalized = model.temporal_weights(td, tp, t0, normalize=True)
        constant_error = float(np.max(np.abs(normalized @ np.ones(len(tp)) - 1)))
        inside = (td >= tp[0] + t0) & (td <= tp[-1] - t0)
        linear_error = (float(np.max(np.abs(normalized[inside] @ tp - td[inside])))
                        if inside.any() else None)
        records.append(dict(t0_days=t0,azimuths=nphi,max_absolute_error=error,
                            constant_error=constant_error,interior_linear_error=linear_error))
        assert error < 3e-8, records[-1]
        assert constant_error < 2e-13, records[-1]
        if linear_error is not None:
            assert linear_error < 2e-10, records[-1]
    face = model.temporal_weights(td, tp, 0.0)
    face_reference = np.column_stack([np.interp(td, tp, impulse) for impulse in np.eye(len(tp))])
    face_error = float(np.max(np.abs(face - face_reference)))
    assert np.all(np.isfinite(face)) and face_error < 2e-13
    return dict(cases=records,face_on_max_absolute_error=face_error,
                method="Direct azimuth quadrature of linearly interpolated unit impulses")


def check_radial():
    """Use astropy's full Planck spectrum, no analytic temperature derivative.

    Independent Gaussian quadrature in log radius additionally tests production
    trapezoid integration and the observer/rest wavelength and flux conversion.
    """
    params = model.DEFAULT.copy()
    params.update(nr=50,nt=10)
    wave = np.array([1180.0, 1928.0, 8897.0])
    radius, edges, temperature, rin, kernel = model.radial_kernels(wave, params)
    wave_rest = wave / (1 + params["redshift"])
    wavelengths = wave_rest[:, None] * units.AA
    flux_density_unit = units.erg / units.s / units.cm**2 / units.AA / units.sr
    nodes, weights = leggauss(128)
    distance_cm = (params["distance_mpc"] * units.Mpc).to_value(units.cm)
    solid_angle_factor = 2 * np.pi * np.cos(np.deg2rad(params["inclination_deg"])) * (rin / distance_cm)**2
    finite_difference = np.empty_like(kernel)
    epsilon = 1e-5
    for k in range(len(radius)):
        low, high = np.log(edges[k:k+2])
        ur = np.exp((low + high) / 2 + (high - low) / 2 * nodes)
        radial_weights = weights * (high - low) / 2 * ur**2
        steady = model.disk_temperature(ur, params)[0]
        fluxes = []
        for sign in [1, -1]:
            temperatures = steady + sign * epsilon * temperature[k]
            spectrum = BlackBody(temperature=temperatures[None, :] * units.K)(wavelengths)
            spectrum = spectrum.to_value(flux_density_unit,
                                          equivalencies=units.spectral_density(wavelengths))
            observed_flux = solid_angle_factor * (spectrum @ radial_weights)
            observed_flux /= (1 + params["redshift"]) * 1e-15
            fluxes.append(observed_flux)
        finite_difference[:, k] = (fluxes[0] - fluxes[1]) / (2 * epsilon)
    relevant = np.abs(kernel) > 1e-8 * np.max(np.abs(kernel))
    convergence = []
    for subdivisions in [128,512,2048]:
        refined_params = dict(params,radial_quadrature=subdivisions)
        tested_kernel = model.radial_kernels(wave,refined_params)[-1]
        relative = np.abs((finite_difference-tested_kernel)[relevant]/tested_kernel[relevant])
        normalized_error = float(np.max(np.abs(finite_difference-tested_kernel))/np.max(np.abs(tested_kernel)))
        convergence.append(dict(radial_subdivisions=subdivisions,
            max_relative_error=float(relative.max()),median_relative_error=float(np.median(relative)),
            max_absolute_error_over_global_kernel_peak=normalized_error))
    # The innermost cell meets T=0 and converges more slowly. Record its
    # fractional error instead of treating a weak individual cell as exact.
    assert convergence[0]["max_absolute_error_over_global_kernel_peak"] < 1e-4
    assert convergence[-1]["max_relative_error"] < 3e-4
    assert convergence[-1]["max_absolute_error_over_global_kernel_peak"] < 1e-6
    for coarse,fine in zip(convergence[:-1],convergence[1:]):
        assert fine["max_absolute_error_over_global_kernel_peak"] < coarse["max_absolute_error_over_global_kernel_peak"]/3
    return dict(convergence=convergence,tested_cells=int(relevant.sum()),gauss_nodes=128,
                default_peak_normalized_tolerance=1e-4,refined_relative_tolerance=3e-4,
                finite_difference_fraction=epsilon,
                method="Astropy full Planck law, independent log-radius Gauss quadrature, central difference")


def check_linear_algebra():
    rng = np.random.default_rng(81)
    t = np.sort(rng.uniform(0, 10, 36))
    bi = np.tile(np.arange(3), 12)
    err = rng.uniform(0.2, 0.7, len(t))
    wave = np.array([1180.0, 1928.0, 8897.0])
    inversion = model.Inversion(t, bi, wave, err,dict(nr=4,nt=8,radial_quadrature=128))
    xi = 10.0
    inversion.factorize(xi)
    flux = rng.normal(size=len(t)) + np.array([20.0,30.0,40.0])[bi]
    fit = inversion.solve(flux)
    aw, bw = inversion.Aw.toarray(), inversion.Bw.toarray()
    design = np.column_stack((aw, bw))
    penalty = np.zeros((design.shape[1],design.shape[1]))
    nq = inversion.A.shape[1]
    penalty[:nq,:nq] = xi * inversion.P.toarray()
    normal = design.T @ design + penalty
    exact = np.linalg.solve(normal, design.T @ (flux / err))
    result = np.r_[fit["q"].T.ravel(),fit["baseline"]]
    solution_error = float(np.max(np.abs(result - exact)))
    assert solution_error < 2e-7, solution_error
    gradient_q = inversion.Aw.T @ (fit["residual"] / err) - xi * inversion.P @ result[:nq]
    gradient_b = inversion.Bw.T @ (fit["residual"] / err)
    gradient = np.r_[gradient_q,gradient_b]
    relative_gradient = float(np.linalg.norm(gradient) / np.linalg.norm(design.T @ (flux / err)))
    assert relative_gradient < 2e-10, relative_gradient

    # All unit data vectors supply the full hat matrix in a tiny exact problem.
    identity = np.eye(len(t))
    q, b = inversion.solve_rhs(inversion.Aw.T @ identity,inversion.Bw.T @ identity)
    hat = inversion.Aw @ q + inversion.Bw @ b
    exact_hat = design @ np.linalg.solve(normal,design.T)
    hat_error = float(np.max(np.abs(hat - exact_hat)))
    assert hat_error < 2e-10, hat_error
    eigenvalues = np.linalg.eigvalsh((hat + hat.T) / 2)
    assert eigenvalues.min() > -2e-10 and eigenvalues.max() < 1 + 2e-10
    exact_edf = float(np.trace(exact_hat))
    probes, seed = 4096, 817
    edf, mcse = inversion.edf(probes=probes,seed=seed)
    independent_rng = np.random.default_rng(seed)
    vectors = independent_rng.choice([-1.0,1.0],size=(len(t),probes))
    trace_samples = np.einsum("ij,ij->j",vectors,exact_hat @ vectors)
    independent_mean = float(trace_samples.mean())
    independent_mcse = float(trace_samples.std(ddof=1) / np.sqrt(probes))
    assert abs(edf - independent_mean) < 2e-10
    assert abs(mcse - independent_mcse) < 2e-10
    assert abs(edf - exact_edf) < 6 * mcse

    # Evaluate the published penalty as scalars, independently of sparse indexing.
    q0 = rng.normal(size=(8,4))
    temp = inversion.temp
    explicit = float(np.sum(q0**2))
    for it in range(8):
        for ir in range(3):
            explicit += (q0[it,ir+1]*temp[ir+1] - q0[it,ir]*temp[ir])**2 / (temp[ir+1]*temp[ir])
    for it in range(7):
        for ir in range(4):
            explicit += (q0[it+1,ir] - q0[it,ir])**2
    quadratic = float(q0.ravel() @ inversion.P @ q0.ravel())
    penalty_error = abs(quadratic - explicit) / explicit
    assert penalty_error < 2e-13
    face_on = model.Inversion(t,bi,wave,err,dict(nr=4,nt=8,inclination_deg=0.0))
    assert np.all(np.isfinite(face_on.A.data))
    return dict(n_data=len(t),n_temperature_parameters=nq,n_free_offsets=3,
                schur_dense_solution_max_absolute_error=solution_error,
                full_normal_equation_relative_residual=relative_gradient,
                hat_matrix_max_absolute_error=hat_error,
                hat_eigenvalue_min=float(eigenvalues.min()),hat_eigenvalue_max=float(eigenvalues.max()),
                exact_effective_parameters=exact_edf,hutchinson_effective_parameters=edf,
                hutchinson_mcse=mcse,hutchinson_probes=probes,
                hutchinson_difference_from_exact=float(edf-exact_edf),
                regularizer_relative_error=float(penalty_error),face_on_forward_finite=True)


def main():
    started = time.perf_counter()
    report = dict(status="running",scope="Synthetic in-memory mathematical verification; no campaign fit")
    try:
        with threadpool_limits(limits=1):
            report["temporal"] = check_temporal()
            report["radial"] = check_radial()
            report["linear_algebra"] = check_linear_algebra()
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        destination = ROOT / "outputs" / "validation.json"
        destination.write_text(json.dumps(report,indent=2),encoding="utf8")
        print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
