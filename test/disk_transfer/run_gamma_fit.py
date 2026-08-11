"""Runnable example: generate one disk response and fit it with MCMC."""

from __future__ import annotations

import numpy as np

from disk_transfer import DiskTransferModel, mdot_from_eddington_ratio
from gamma_mcmc import fit_shifted_gamma_mcmc, print_summary, save_results


# 1. Fixed grids. Use a coarser spatial grid first to verify the workflow.
time_days = np.linspace(0.0, 30.0, 601)  # 0.05 day per bin
wavelengths_angstrom = np.array(
    [1500.0, 2500.0, 3500.0, 5000.0]
)

model = DiskTransferModel(
    wavelengths_angstrom=wavelengths_angstrom,
    time_days=time_days,
    global_rmin_rg=6.0,
    rout_rg=1.0e4,
    nr=900,       # increase toward 900 for final calculations
    nphi=720,     # increase toward 720 for final calculations
    pulse_width_days=0.05,
)

# 2. Physical parameters for one forward-model evaluation.
m_bh_msun = 1.0e8
mdot_g_s = mdot_from_eddington_ratio(
    m_bh_msun=m_bh_msun,
    eddington_ratio=0.10,
    radiative_efficiency=0.10,
)

prediction = model.predict(
    m_bh_msun=m_bh_msun,
    mdot_g_s=mdot_g_s,
    lx_erg_s=1.0e44,
    h_rg=10.0,
    inclination_deg=30.0,
    rin_rg=6.0,
)

# 3. 分别拟合所有波长
all_results = {}

for wavelength_index, wavelength in enumerate(
    wavelengths_angstrom
):
    print("\n" + "=" * 60)
    print(f"开始拟合 wavelength = {wavelength:.1f} Angstrom")
    print("=" * 60)

    response = prediction["pulsed_response"][
        wavelength_index
    ]

    result = fit_shifted_gamma_mcmc(
        prediction["time_days"],
        response,
        nwalkers=32,
        nsteps=4000,
        burn=1500,
        thin=5,
        random_seed=1234 + wavelength_index,
        progress=True,
    )

    all_results[wavelength] = result

    print(
        f"Wavelength = {wavelength:.1f} Angstrom"
    )
    print_summary(result)

    output_directory = (
        f"gamma_fit_output/"
        f"{wavelength:.0f}_angstrom"
    )

    save_results(
        result,
        output_directory,
    )

    print(
        f"结果保存到 ./{output_directory}/"
    )
response = prediction["pulsed_response"][wavelength_index]

# 4. Run MCMC. For a quick test, use nsteps=1000 and burn=300.
result = fit_shifted_gamma_mcmc(
    prediction["time_days"],
    response,
    nwalkers=32,
    nsteps=4000,
    burn=1500,
    thin=5,
    random_seed=1234,
    progress=True,
)

# 5. Print and save outputs.
print(f"Wavelength = {wavelengths_angstrom[wavelength_index]:.1f} Angstrom")
print_summary(result)
save_results(result, "gamma_fit_output")
print("\nFiles written to ./gamma_fit_output/")
