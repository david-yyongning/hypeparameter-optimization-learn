import numpy as np
import optuna

class SimulatedAnnealingSampler(optuna.samples.BaseSampler):
    def __init__(self, temperature=100, init_trial=None, restart_threshould=50):
        self._rng = np.random.RandomState()
        self._temperature = temperature
        self._current_trial = init_trial  # Maintains the best.
        self._counter = 0 # counter for iterations without improvement
        self._restart_threshold = restart_threshould # threshold for restart
        
    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}
        
        # Simulated Annealing (SA) algorithm
        # 1. Calcuate transition probability
        #   If the current trial is None or the previous trial is better than the current trial,
        #    the next trial is the previous trial.
        #   Otherwise, the next trial is the previous trial with a certain probability.
        prev_trial = study.trials[-2]
        if self._current_trial is None or prev_trial.value <= self._current_trial.value:
            # Accept s* if E(s*)<=E(s) but may not always based on "Barrier avoidance principle"
            probability = 1.0
        else:
            # Accept s* with a certain probability 𝑒^(−(𝑒^∗−𝑒)/𝑇)
            probability = np.exp((self._current_trial.value - prev_trial.value) / self._temperature)
            
        self._temperature *= 0.9 # Decrease temperature, hyperparameter to be tuned.
        
        # 2. Transit current state, either to new state or remain unchanged
        # If the probability is higher than a random number, the next trial is the previous trial.
        if probability >= self._rng.uniform(0, 1):
            self._current_trial = prev_trial
            
        # 2.1. Restart the search if no improvement is observed for a certain number of iterations.
        if self._current_trial.value > study.best_trial.value:
            self._counter += 1
            if self._counter >= self._restart_threshold:
                self._current_trial = study.best_trial
                self._counter = 0
                print(f"Restart the search at trial {trial.number}")
        else:
            self._counter = 0
            
        # 3. Sample parameters from the search space (neightborhood of the current trial)
        params = {}
        for param_name, param_distribution in search_space.items():
            
            # Only support suggest_float() with `step` `None` or 1.0, and `log` `False`
            if (not isinstance(param_distribution, optuna.distributions.FloatDistribution)
                or (param_distribution.step is not None and param_distribution.step != 1)
                or param_distribution.log):
                msg = "Only suggest_float() with `step` `None` or 1.0 and `log` `False` is supported"
                raise NotImplementedError(msg)
            
            # Sample parameters from the neighborhood of the current trial
            current_value = self._current_trial.params[param_name]
            # width 10% of the range, small enough? Principle: sufficiently near neighbor or diameter of the search graph must be small
            width = (param_distribution.high - param_distribution.low) * 0.1
            neighbor_low = max(current_value - width, param_distribution.low)
            neighbor_high = min(current_value + width, param_distribution.high)
            params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)
            
        return params
    
        
    

    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_independent(self, study, trial, param_name, param_distribution):
        if (
            not isinstance(param_distribution, optuna.distributions.FloatDistribution)
            or (param_distribution.step is not None and param_distribution.step != 1)
            or param_distribution.log
        ):
            msg = (
                "Only suggest_float() with `step` `None` or 1.0 and"
                " `log` `False` is supported"
            )
            raise NotImplementedError(msg)

        return self._rng.uniform(param_distribution.low, param_distribution.high)