# %%
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize

# for plotting if needed
import matplotlib
import matplotlib.pyplot as plt

from undi_tools.atoms import build_undi_neighbors
from undi_tools.atoms import complete_undi_neighbors
from undi_tools.atoms_scaling import scale_neighbors
from undi_tools.atoms_scaling import only_muon_cluster
from undi_tools.atoms_scaling import partition_into_shells

# %%
try:
    import lmfit
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False

# %%
def powder_celio_signal_generic(neighbors, tlist, **celio_kwargs):
    """
    Powder-averaged P_z(t) using Celio's method along 3 orthogonal directions[cite: 4].
    Accepts arbitrary arguments (**celio_kwargs) for undi configuration.
    """
    try:
        from undi import MuonNuclearInteraction
    except (ImportError, ModuleNotFoundError):
        import sys
        if 'undi_path' in celio_kwargs:
            sys.path.append(celio_kwargs.pop('undi_path'))
        from undi import MuonNuclearInteraction

    # Extract specific kwargs or fallback to default Celio settings[cite: 4]
    k = celio_kwargs.get('k', 1)
    nrep = celio_kwargs.get('nrep', 1)
    single_precision = celio_kwargs.get('single_precision', True)
    algorithm = celio_kwargs.get('algorithm', 'fast')
    log_level = celio_kwargs.get('log_level', 'warning')

    signal = np.zeros(len(tlist))
    # 3-orthogonal directions powder average trick[cite: 4]
    for direction in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        NS = MuonNuclearInteraction(deepcopy(neighbors), log_level=log_level)
        NS.translate_rotate_sample_vec(direction)
        for _ in range(nrep):
            signal += NS.celio_on_steroids(
                tlist, k=k, single_precision=single_precision, progress=False, algorithm=algorithm
            )
    return signal / (3.0 * nrep)

# %%
class WilkinsonClusterBuilder:
    """
    Generic cluster manager: handles unperturbed structure loading,
    partitioning into Muon, NN, and NNN shells, and rescale dynamics.
    """
    def __init__(
        self, 
        atoms, 
        muon_pos, 
        nn_cutoffs, 
        nnn_cutoffs, 
        Q_overrides=None, 
    ):
        self.atoms = atoms
        self.muon_pos = np.array(muon_pos)
        
        # Build base NN cluster
        nn_raw = build_undi_neighbors(
            atoms, self.muon_pos, cutoffs=nn_cutoffs, inf_cutoff=10.0,
            quadrupole_moments_overrides=Q_overrides, verbose_neighbors=False,
        )
        self.nn_cluster = complete_undi_neighbors(nn_raw, log_level="")

        # Build base NNN cluster
        nnn_raw = build_undi_neighbors(
            atoms, self.muon_pos, cutoffs=nnn_cutoffs, inf_cutoff=10.0,
            quadrupole_moments_overrides=Q_overrides, verbose_neighbors=False,
        )
        self.nnn_cluster = complete_undi_neighbors(nnn_raw, log_level="")

        # Partition into disjoint shells
        self.mu_atoms = only_muon_cluster(self.nn_cluster)
        
        nn_shells = partition_into_shells(self.mu_atoms[0], [self.mu_atoms, self.nn_cluster], rtol=1e-4)
        self.nn_bare = nn_shells[1]

        nnn_shells = partition_into_shells(self.mu_atoms[0], [self.nn_cluster, self.nnn_cluster], rtol=1e-4)
        self.nnn_bare = nnn_shells[1]

    def build_rescaled_cluster(self, r_nn=None, zeta=None, max_nnn_atoms=None):
        """
        Rescales the nearest-neighbor distance (r_nn in Angstroms) 
        and distant NNN interactions (zeta).
        """
        cluster = deepcopy(self.mu_atoms)

        # 1. Scale Nearest-Neighbors (r_nn in Angstroms)
        nn_scaled = deepcopy(self.nn_bare)
        if r_nn is not None:
            ## scale nn by r_nn
            # nnn_scaled = scale_neighbors(
            #     nn_scaled, scale=float(r_nn), mode="radius", include_muon=False
            # )
            ## or with full illustrations
            for atom in nn_scaled:
                r_orig = np.linalg.norm(atom['Position'])
                if r_orig > 0:
                    # Scale vector magnitude to target r_nn (converted to meters)
                    atom['Position'] = (atom['Position'] / r_orig) * (r_nn * 1e-10)

        cluster += nn_scaled

        # 2. Scale Next-Nearest-Neighbors (zeta)
        nnn_scaled = deepcopy(self.nnn_bare)
        if max_nnn_atoms is not None:
            nnn_scaled = nnn_scaled[:max_nnn_atoms]

        if zeta is not None:
            ## Scale coordinates by zeta so dipolar field scales by zeta^{-3}
            # nnn_scaled = scale_neighbors(
            #     nnn_scaled, scale=float(zeta), mode="multiplicative", include_muon=False
            # )
            ## or with full illustrations
            for atom in nnn_scaled:
                atom["Position"] = atom["Position"] * float(zeta)
            
        cluster += nnn_scaled
        
        return cluster

# %%
class WilkinsonMuonPolarizationFitter:
    """
    Generic model fitter supporting custom signal parameters, flexible 
    parameter constraints, dual lmfit/scipy backends, fit metrics (R²), 
    and plotting using a dense NumPy grid.
    
    Fixed: Uses derivative-free Nelder-Mead optimization to handle stochastic 
    noise from Celio simulations without gradient failure.
    """
    def __init__(
        self, 
        cluster_builder, 
        tlist, 
        y_data, 
        err_data=None, 
        method='nelder',
        max_nnn_atoms=None, 
        **celio_kwargs
    ):
        self.builder = cluster_builder
        self.t = np.asarray(tlist)
        self.y = np.asarray(y_data)
        self.err = np.asarray(err_data) if err_data is not None else None
        self.weights = 1.0 / self.err if self.err is not None else np.ones_like(self.y)
        self.method = method
        self.max_nnn_atoms = max_nnn_atoms
        
        # Store arbitrary signal calculation parameters
        self.celio_kwargs = celio_kwargs
        self._signal_cache = {}
        self.last_fit = None


    def simulate_powder_for_only_rnn_zeta(self, t, r_nn, zeta):
            """
            Calculates polarization and caches results based on (r_nn, zeta), 
            t parameters, and current celio_kwargs configuration.
            """
            t = np.asarray(t)
            # Include time vector dimensions and range in the cache key
            t_info = (len(t), float(t[0]), float(t[-1]))
            kwargs_tuple = tuple(sorted((k, str(v)) for k, v in self.celio_kwargs.items()))
            
            cache_key = (round(float(r_nn), 4), round(float(zeta), 4), t_info, kwargs_tuple)

            if cache_key not in self._signal_cache:
                cluster = self.builder.build_rescaled_cluster(
                    r_nn=r_nn, zeta=zeta, max_nnn_atoms=self.max_nnn_atoms
                )
                self._signal_cache[cache_key] = powder_celio_signal_generic(
                    cluster, t, **self.celio_kwargs
                )
            return self._signal_cache[cache_key]


    # def simulate_powder_for_only_rnn_zeta(self, t, r_nn, zeta):
    #     """
    #     Calculates polarization and caches results based on (r_nn, zeta)
    #     and current celio_kwargs configuration.
    #     """
    #     # Form a unique cache key including celio_kwargs settings
    #     kwargs_tuple = tuple(sorted((k, str(v)) for k, v in self.celio_kwargs.items()))
    #     cache_key = (round(float(r_nn), 4), round(float(zeta), 4), kwargs_tuple)

    #     if cache_key not in self._signal_cache:
    #         cluster = self.builder.build_rescaled_cluster(
    #             r_nn=r_nn, zeta=zeta, max_nnn_atoms=self.max_nnn_atoms
    #         )
    #         # Unpack stored celio parameters into the signal generator
    #         self._signal_cache[cache_key] = powder_celio_signal_generic(
    #             cluster, t, **self.celio_kwargs
    #         )
    #     return self._signal_cache[cache_key]

    def _model_eval(self, t, r_nn, zeta, amplitude, background):
        P_z = self.simulate_powder_for_only_rnn_zeta(t, r_nn, zeta)
        return amplitude * P_z + background

    def fit(self, param_config=None, method=None, force_scipy=False):
        """Fits the signal using Nelder-Mead (derivative-free) optimization."""
        default_config = {
            'r_nn':       {'value': 1.18, 'min': 0.8, 'max': 1.5, 'vary': True},
            'zeta':       {'value': 0.937, 'min': 0.5, 'max': 1.2, 'vary': True},
            'amplitude':  {'value': 0.20, 'min': 0.0, 'max': 1.0, 'vary': True},
            'background': {'value': 0.00, 'min': -0.1, 'max': 0.5, 'vary': True}
        }

        config = default_config.copy()
        if param_config is not None:
            for p, settings in param_config.items():
                if p in config:
                    config[p].update(settings)

        self.method = method if method is not None else self.method

        if HAS_LMFIT and not force_scipy:
            print(f"--> Fitting using lmfit ({self.method})...")
            model = lmfit.Model(self._model_eval, independent_vars=['t'])
            params = model.make_params()
            
            for p_name, p_opts in config.items():
                params.add(
                    p_name,
                    value=p_opts['value'],
                    min=p_opts['min'],
                    max=p_opts['max'],
                    vary=p_opts['vary']
                )

            # Use Nelder-Mead simplex algorithm to handle stochastic noise
            res = model.fit(self.y, params, t=self.t, weights=self.weights, method=self.method)
            final_params = res.values
            y_fit = res.best_fit
            engine = "lmfit"

        else:
            print("--> Fitting using SciPy minimize (Nelder-Mead)...")
            engine = "scipy"
            p_names = ['r_nn', 'zeta', 'amplitude', 'background']
            active_names = [p for p in p_names if config[p]['vary']]
            fixed_values = {p: config[p]['value'] for p in p_names if not config[p]['vary']}
            
            x0 = [config[p]['value'] for p in active_names]
            bounds = [(config[p]['min'], config[p]['max']) for p in active_names]

            def scipy_objective(active_vals):
                current_params = fixed_values.copy()
                for name, val in zip(active_names, active_vals):
                    current_params[name] = val
                
                # Soft boundary penalty
                for name, val in current_params.items():
                    if config[name]['vary']:
                        if val < config[name]['min'] or val > config[name]['max']:
                            return 1e6

                y_model = self._model_eval(
                    self.t,
                    current_params['r_nn'],
                    current_params['zeta'],
                    current_params['amplitude'],
                    current_params['background']
                )
                residuals = (self.y - y_model) * self.weights
                return np.sum(residuals**2)

            res = minimize(
                scipy_objective,
                x0=x0,
                method='Nelder-Mead',
                bounds=bounds,
                options={'xatol': 1e-3, 'fatol': 1e-3}
            )

            final_params = fixed_values.copy()
            for name, val in zip(active_names, res.x):
                final_params[name] = val

            y_fit = self._model_eval(
                self.t,
                final_params['r_nn'],
                final_params['zeta'],
                final_params['amplitude'],
                final_params['background']
            )

        # Calculate R-squared metric
        ss_res = np.sum((self.y - y_fit) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        self.last_fit = {
            "engine": engine,
            "result": res,
            "params": final_params,
            'builder': self.builder,
            'x_dat': self.t,
            'y_dat': self.y,
            'y_err': self.err,
            'x_fit': self.t,
            "y_fit": y_fit,
            "r2": r2,
            "config": config
        }

        self.print_summary()
        return self.last_fit

    def print_summary(self):
        """Prints a summary table of parameter results and fit metrics."""
        if self.last_fit is None:
            print("No fit has been executed yet.")
            return

        p = self.last_fit["params"]
        cfg = self.last_fit["config"]
        r2 = self.last_fit["r2"]
        engine = self.last_fit["engine"]

        print("\n" + "="*55)
        print(f" FIT RESULTS SUMMARY ({engine.upper()}) ")
        print("="*55)
        print(f"{'Parameter':<12} | {'Value':<10} | {'Status':<8} | {'Bounds':<15}")
        print("-" * 55)
        for name in ['r_nn', 'zeta', 'amplitude', 'background']:
            val = p[name]
            status = "Vary" if cfg[name]['vary'] else "Fixed"
            b_str = f"[{cfg[name]['min']}, {cfg[name]['max']}]" if cfg[name]['vary'] else "N/A"
            print(f"{name:<12} | {val:<10.5f} | {status:<8} | {b_str:<15}")
        print("-" * 55)
        print(f"R-squared (R²): {r2:.6f}")
        print("="*55 + "\n")

    def plot_fit(self, t_grid=None, title="Muon Polarization Fit"):
        """Plots experimental data against the fitted model curve."""
        if self.last_fit is None:
            raise RuntimeError("Run .fit() before plotting results.")

        p = self.last_fit["params"]

        if t_grid is None:
            t_grid = np.linspace(self.t.min(), self.t.max(), 500)
        else:
            t_grid = np.asarray(t_grid)

        y_grid = self._model_eval(
            t_grid, p['r_nn'], p['zeta'], p['amplitude'], p['background']
        )

        plt.figure(figsize=(8, 5))

        if self.err is not None:
            plt.errorbar(
                self.t * 1e6, self.y, yerr=self.err, fmt='o', color='black',
                ecolor='lightgray', elinewidth=1, capsize=2, label='Data', alpha=0.7
            )
        else:
            plt.plot(self.t * 1e6, self.y, 'o', color='black', label='Data', alpha=0.7)

        plt.plot(
            t_grid * 1e6, y_grid, '-', color='red', linewidth=2,
            label=f"Fit ($R^2={self.last_fit['r2']:.4f}$)"
        )

        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel(r'Time ($\mu$s)', fontsize=11)
        plt.ylabel(r'Asymmetry / $P_z(t)$', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# UNDER DEVELOPMENTS

# %%
class WilkinsonClusterBuilder2:
    """
    Generic cluster manager: partitions structures into individual distance-based 
    shells and rescales arbitrary sets of r_nn and r_nnn/zeta parameters.
    """
    def __init__(
        self, 
        atoms, 
        muon_pos, 
        nn_cutoffs, 
        nnn_cutoffs, 
        Q_overrides=None, 
    ):
        self.atoms = atoms
        self.muon_pos = np.array(muon_pos)
        
        # Build base NN & NNN raw clusters
        nn_raw = build_undi_neighbors(
            atoms, self.muon_pos, cutoffs=nn_cutoffs, inf_cutoff=10.0,
            quadrupole_moments_overrides=Q_overrides, verbose_neighbors=False,
        )
        self.nn_cluster = complete_undi_neighbors(nn_raw, log_level="")

        nnn_raw = build_undi_neighbors(
            atoms, self.muon_pos, cutoffs=nnn_cutoffs, inf_cutoff=10.0,
            quadrupole_moments_overrides=Q_overrides, verbose_neighbors=False,
        )
        self.nnn_cluster = complete_undi_neighbors(nnn_raw, log_level="")

        # Partition into disjoint shells
        self.mu_atoms = only_muon_cluster(self.nn_cluster)
        
        nn_shells = partition_into_shells(self.mu_atoms[0], [self.mu_atoms, self.nn_cluster], rtol=1e-4)
        self.nn_bare = nn_shells[1]

        nnn_shells = partition_into_shells(self.mu_atoms[0], [self.nn_cluster, self.nnn_cluster], rtol=1e-4)
        self.nnn_bare = nnn_shells[1]

    def _group_by_distance(self, atom_list, rtol=1e-3):
        """Groups atoms into distinct radial shells based on distance from muon."""
        shells = []
        for atom in atom_list:
            r = np.linalg.norm(atom['Position'])
            matched = False
            for shell in shells:
                if np.isclose(r, shell['r_orig'], rtol=rtol):
                    shell['atoms'].append(atom)
                    matched = True
                    break
            if not matched:
                shells.append({'r_orig': r, 'atoms': [atom]})
        
        # Sort shells by ascending distance
        shells.sort(key=lambda s: s['r_orig'])
        return shells

    def build_rescaled_cluster(self, r_nn=None, r_nnn=None, zeta=None, max_nnn_atoms=None):
        """
        Rescales arbitrary numbers of NN and NNN shell distances.
        
        Parameters:
        -----------
        r_nn : float, list, or dict
            - float: scales first shell or all NN uniformly.
            - list/tuple: [r_nn1, r_nn2, ...] maps sequentially to sorted NN shells.
            - dict: {'r_nn1': val, 'r_nn2': val} or {shell_index: val}.
        zeta : float, list, or dict
            Multiplicative scaling factor(s) for distant/NNN interactions.
        """
        cluster = deepcopy(self.mu_atoms)

        # 1. Scale Nearest-Neighbors
        nn_scaled = deepcopy(self.nn_bare)
        if r_nn is not None:
            nn_shells = self._group_by_distance(nn_scaled)
            
            # Convert r_nn input into a list matching shell indices
            if isinstance(r_nn, (int, float)):
                r_nn_list = [r_nn] * len(nn_shells)
            elif isinstance(r_nn, dict):
                r_nn_list = [r_nn.get(f"r_nn{i+1}", r_nn.get(i, shell['r_orig']*1e10)) 
                             for i, shell in enumerate(nn_shells)]
            else:
                r_nn_list = list(r_nn)

            for idx, shell in enumerate(nn_shells):
                if idx < len(r_nn_list) and r_nn_list[idx] is not None:
                    target_r = r_nn_list[idx] * 1e-10  # Angstroms to meters
                    for atom in shell['atoms']:
                        r_orig = np.linalg.norm(atom['Position'])
                        if r_orig > 0:
                            atom['Position'] = (atom['Position'] / r_orig) * target_r

        cluster += nn_scaled

        # 2. Scale Next-Nearest-Neighbors (NNN)
        nnn_scaled = deepcopy(self.nnn_bare)
        if max_nnn_atoms is not None:
            nnn_scaled = nnn_scaled[:max_nnn_atoms]

        if zeta is not None or r_nnn is not None:
            nnn_shells = self._group_by_distance(nnn_scaled)
            
            # Handle direct distance scaling (r_nnn) or factor scaling (zeta)
            if r_nnn is not None:
                r_nnn_list = [r_nnn] if isinstance(r_nnn, (int, float)) else list(r_nnn)
                for idx, shell in enumerate(nnn_shells):
                    if idx < len(r_nnn_list) and r_nnn_list[idx] is not None:
                        target_r = r_nnn_list[idx] * 1e-10
                        for atom in shell['atoms']:
                            r_orig = np.linalg.norm(atom['Position'])
                            if r_orig > 0:
                                atom['Position'] = (atom['Position'] / r_orig) * target_r
            
            elif zeta is not None:
                zeta_list = [zeta] if isinstance(zeta, (int, float)) else list(zeta)
                for idx, shell in enumerate(nnn_shells):
                    scale_fac = zeta_list[idx] if idx < len(zeta_list) else zeta_list[-1]
                    for atom in shell['atoms']:
                        atom['Position'] = atom['Position'] * float(scale_fac)

        cluster += nnn_scaled
        return cluster

# %%
class WilkinsonMuonPolarizationFitter2:
    """
    Generic model fitter supporting arbitrary dynamic parameter configurations 
    (multiple r_nn_i, r_nnn_i, zeta_i), signal caching, dual backends, and fit metrics.
    """
    def __init__(
        self, 
        cluster_builder, 
        tlist, 
        y_data, 
        err_data=None, 
        max_nnn_atoms=None, 
        method='nelder',
        **celio_kwargs
    ):
        self.builder = cluster_builder
        self.t = np.asarray(tlist)
        self.y = np.asarray(y_data)
        self.err = np.asarray(err_data) if err_data is not None else None
        self.weights = 1.0 / self.err if self.err is not None else np.ones_like(self.y)
        self.max_nnn_atoms = max_nnn_atoms

        self.method = method
        
        self.celio_kwargs = celio_kwargs
        self._signal_cache = {}
        self.last_fit = None

    def simulate_powder_generic(self, t, **geom_params):
        """
        Calculates polarization and caches results based on generic structural parameters.
        """
        t = np.asarray(t)
        t_info = (len(t), float(t[0]), float(t[-1]))
        
        # Build cache key from sorted geometry parameters and celio settings
        geom_tuple = tuple(sorted((k, round(float(v), 5)) for k, v in geom_params.items()))
        kwargs_tuple = tuple(sorted((k, str(v)) for k, v in self.celio_kwargs.items()))
        cache_key = (geom_tuple, t_info, kwargs_tuple)

        if cache_key not in self._signal_cache:
            # Separate geometry kwargs into r_nn, r_nnn, and zeta groupings
            r_nn_dict = {k: v for k, v in geom_params.items() if k.startswith('r_nn') and not k.startswith('r_nnn')}
            r_nnn_dict = {k: v for k, v in geom_params.items() if k.startswith('r_nnn')}
            zeta_dict = {k: v for k, v in geom_params.items() if k.startswith('zeta')}

            cluster = self.builder.build_rescaled_cluster(
                r_nn=r_nn_dict if r_nn_dict else None,
                r_nnn=r_nnn_dict if r_nnn_dict else None,
                zeta=zeta_dict if zeta_dict else None,
                max_nnn_atoms=self.max_nnn_atoms
            )
            self._signal_cache[cache_key] = powder_celio_signal_generic(
                cluster, t, **self.celio_kwargs
            )
        return self._signal_cache[cache_key]

    def _model_eval(self, t, **params):
        """Dynamic evaluation wrapper dividing parameters into signal vs scale/background."""
        amplitude = params.pop('amplitude', 1.0)
        background = params.pop('background', 0.0)
        
        P_z = self.simulate_powder_generic(t, **params)
        return amplitude * P_z + background

    def fit(self, param_config, method=None, force_scipy=False):
        """
        Fits asymmetry using user provided param_config dictionary containing 
        arbitrary structural and scaling parameters (e.g., r_nn1, r_nn2, r_nnn1, zeta4).
        """
        config = deepcopy(param_config)
        p_names = list(config.keys())
        geom_names = [p for p in p_names if p not in ['amplitude', 'background']]
        self.method = method if method is not None else self.method

        if HAS_LMFIT and not force_scipy:
            print(f"--> Fitting using lmfit ({self.method})...")
            
            def lmfit_wrapper(t, **kwargs):
                return self._model_eval(t, **kwargs)

            model = lmfit.Model(lmfit_wrapper, independent_vars=['t'])
            params = model.make_params()
            
            for p_name, p_opts in config.items():
                params.add(
                    p_name,
                    value=p_opts['value'],
                    min=p_opts.get('min', -np.inf),
                    max=p_opts.get('max', np.inf),
                    vary=p_opts.get('vary', True)
                )

            res = model.fit(self.y, params, t=self.t, weights=self.weights, method=self.method)
            final_params = res.values
            y_fit = res.best_fit
            engine = "lmfit"

        else:
            print("--> Fitting using SciPy minimize (Nelder-Mead)...")
            engine = "scipy"
            active_names = [p for p in p_names if config[p].get('vary', True)]
            fixed_values = {p: config[p]['value'] for p in p_names if not config[p].get('vary', True)}
            
            x0 = [config[p]['value'] for p in active_names]
            bounds = [(config[p].get('min', -np.inf), config[p].get('max', np.inf)) for p in active_names]

            def scipy_objective(active_vals):
                current_params = fixed_values.copy()
                for name, val in zip(active_names, active_vals):
                    current_params[name] = val
                
                # Soft boundary penalty
                for name in active_names:
                    val = current_params[name]
                    p_min = config[name].get('min', -np.inf)
                    p_max = config[name].get('max', np.inf)
                    if val < p_min or val > p_max:
                        return 1e6

                y_model = self._model_eval(self.t, **current_params)
                residuals = (self.y - y_model) * self.weights
                return np.sum(residuals**2)

            res = minimize(
                scipy_objective,
                x0=x0,
                method='Nelder-Mead',
                bounds=bounds,
                options={'xatol': 1e-3, 'fatol': 1e-3}
            )

            final_params = fixed_values.copy()
            for name, val in zip(active_names, res.x):
                final_params[name] = val

            y_fit = self._model_eval(self.t, **final_params)

        # Metrics
        ss_res = np.sum((self.y - y_fit) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        self.last_fit = {
            "engine": engine,
            "result": res,
            "params": final_params,
            'x_fit': self.t,
            "y_fit": y_fit,
            "r2": r2,
            "config": config
        }

        self.print_summary()
        return self.last_fit

    def print_summary(self):
        """Prints dynamically structured summary table."""
        if self.last_fit is None:
            print("No fit executed yet.")
            return

        p = self.last_fit["params"]
        cfg = self.last_fit["config"]
        r2 = self.last_fit["r2"]
        engine = self.last_fit["engine"]

        print("\n" + "="*58)
        print(f" FIT RESULTS SUMMARY ({engine.upper()}) ")
        print("="*58)
        print(f"{'Parameter':<14} | {'Value':<10} | {'Status':<8} | {'Bounds':<16}")
        print("-" * 58)
        for name, val in p.items():
            status = "Vary" if cfg[name].get('vary', True) else "Fixed"
            b_str = f"[{cfg[name].get('min')}, {cfg[name].get('max')}]" if status == "Vary" else "N/A"
            print(f"{name:<14} | {val:<10.5f} | {status:<8} | {b_str:<16}")
        print("-" * 58)
        print(f"R-squared (R²): {r2:.6f}")
        print("="*58 + "\n")