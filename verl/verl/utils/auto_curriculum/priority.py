from typing import Any
import numpy as np
from collections import defaultdict
import random
import torch
import copy
import verl.utils.torch_functional as verl_F

def get_bandit_priority(*args, **kwargs):
    assert 'type' in kwargs, 'type is required for initializing bandit priority'
    type = kwargs['type']
    if type == 'joint':
        return JointBandit(*args, **kwargs)
    elif type == 'independent':
        return IndependentBandit(*args, **kwargs)
    else:
        raise ValueError(f'Invalid type: {type}')

class BasePriority:

    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DifficultyPriority(BasePriority):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data: dict[str, Any]):
        return data['extra_info']['difficulty']


class ReverseDifficultyPriority(BasePriority):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data: dict[str, Any]):
        return -data['extra_info']['difficulty']


class EntropyPriority(BasePriority):
    """EDCO-style perplexity-based priority. Higher perplexity = higher priority (sampled first).

    Stores per-token average negative log-prob (i.e. perplexity without exp).
    Since exp is monotonic, sorting by this value is equivalent to sorting by perplexity.
    Values are updated periodically by compute_prefix_entropy() in the trainer.
    Initially, all samples have equal (random) priority.
    """

    def __init__(self, dataset_size, **kwargs):
        self.dataset_size = dataset_size
        self.entropy_values = np.zeros(dataset_size)  # initialized to 0; updated at first entropy recompute

    def __call__(self, data: dict[str, Any]):
        idx = data['extra_info'].get('index', 0)
        return self.entropy_values[idx]

    def update_entropy(self, entropy_dict: dict):
        """Update entropy values from compute_prefix_entropy result.

        Args:
            entropy_dict: dict mapping sample index -> entropy value
        """
        for idx, entropy in entropy_dict.items():
            self.entropy_values[int(idx)] = entropy

    def get_metrics(self):
        valid = self.entropy_values[self.entropy_values != 0]
        if len(valid) == 0:
            return {}
        return {
            'ppl/mean': float(np.mean(valid)),
            'ppl/std': float(np.std(valid)),
            'ppl/max': float(np.max(valid)),
            'ppl/min': float(np.min(valid)),
        }

class BanditPriority(BasePriority):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.temperature = kwargs.get('temperature', 1.0)
        self.learning_rate = kwargs.get('lr', 0.1)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.method = kwargs.get('method', 'boltzmann')
        self.objective = kwargs.get('objective', 'adv')
        self.feature = kwargs.get('feature', ['difficulty'])

        if type(self.feature) == str:
            self.feature = [self.feature]


        assert 'dataset' in kwargs, 'dataset is required for initializing bandit priority'
        
        self.init_bandit(kwargs['dataset'])

        self.q_values = np.zeros(self.num_arms)
    
    def __call__(self, data: dict[str, Any]):
        return random.random()



    def get_metrics(self):
        q_vals = {f'bandit/{group}_q_value': self.q_values[self.group_to_arm[group]] for group in self.group_to_arm}
        probs = {f'bandit/{group}_prob': self.get_probs(self.method)[self.group_to_arm[group]] for group in self.group_to_arm}
        return {**q_vals, **probs}



    def init_bandit(self, dataset):

        self.idx_to_group = {idx: self.data_to_group(item) for idx, item in enumerate(dataset)}
        self.group_to_idx = {group: [] for group in self.idx_to_group.values()}
        for idx, group in self.idx_to_group.items():
            self.group_to_idx[group].append(idx)
        self.arm_to_group = {i: key for i, key in enumerate(self.group_to_idx.keys())}
        self.group_to_arm = {key: i for i, key in enumerate(self.group_to_idx.keys())}
        self.num_arms = len(self.group_to_idx)

        self.arm_to_idx = {i: [] for i in range(self.num_arms)}
        for group, arm in self.group_to_arm.items():
            self.arm_to_idx[arm].extend(self.group_to_idx[group])

        if self.objective == 'progress':
            self.group_to_last_reward = {group: 0 for group in self.group_to_idx.keys()}

    def get_probs(self, method: str):
        if method == 'boltzmann':
            return np.exp(self.q_values / self.temperature) / np.sum(np.exp(self.q_values / self.temperature))
        elif method == 'normalized_q':
            assert all(self.q_values >= 0), 'q_values should be non-negative for normalized_q method'
            clip_q_values = np.maximum(self.q_values, 0.01)
            probs = clip_q_values / np.sum(clip_q_values)
            return probs
        elif method == 'greedy':
            return np.exp(self.q_values / self.temperature) / np.sum(np.exp(self.q_values / self.temperature))
        else:
            raise ValueError(f'Invalid sample method: {method}')


    def sample_arms(self, num_samples: int):
        if self.method == 'boltzmann':
            probs = self.get_probs('boltzmann')
            return np.random.choice(self.num_arms, size=num_samples, p=probs)
        elif self.method == 'normalized_q':
            probs = self.get_probs('normalized_q')
            return np.random.choice(self.num_arms, size=num_samples, p=probs)
        elif self.method == 'greedy':
            arms = []
            for i in range(num_samples):
                if random.random() < self.epsilon:
                    max_arm = np.argmax(self.q_values)
                    arm_to_sample = random.randint(0, self.num_arms - 1)
                    while arm_to_sample == max_arm:
                        arm_to_sample = random.randint(0, self.num_arms - 1)
                    arms.append(arm_to_sample)
                else:
                    arms.append(np.argmax(self.q_values))
            return np.array(arms)
        else:
            raise ValueError(f'Invalid sample method: {self.method}')

    def update(self, *args, **kwargs):
        assert 'data_info' in kwargs, 'data_info is required for updating bandit priority'
        assert 'adv' in kwargs, 'adv is required for updating bandit priority'

        response_mask = kwargs['response_mask']


        data_info = kwargs['data_info']
        if self.objective in ['adv', 'unnormalized_adv']:
            if self.objective == 'adv':
                obj = kwargs['adv']
            else:
                obj = kwargs['unnormalized_adv']

            obj = verl_F.masked_mean(obj.abs(), response_mask, axis=-1)

            group_to_obj = defaultdict(list)
            for data, obj_val in zip(data_info, obj):
                group = self.extra_info_to_group(data)
                group_to_obj[group].append(obj_val.item())
            for group, obj_vals in group_to_obj.items():
                group_to_obj[group] = np.mean(np.abs(obj_vals)) # New Q-value is the mean of the absolute advantages


        elif self.objective == 'progress':
            assert 'rewards' in kwargs, 'rewards is required for updating bandit priority'
            rewards = kwargs['rewards']
            sequence_reward = rewards.sum(-1)
            group_to_reward = defaultdict(list)
            for data, reward in zip(data_info, sequence_reward):
                group = self.extra_info_to_group(data)
                group_to_reward[group].append(reward.item())

            
            group_to_obj = {}
            for group, rewards in self.group_to_last_reward.items():
                if group in group_to_reward:
                    group_to_reward[group] = np.mean(group_to_reward[group])
                else:
                    group_to_reward[group] = rewards
                group_to_obj[group] = group_to_reward[group] - self.group_to_last_reward[group]
                self.group_to_last_reward[group] = group_to_reward[group]
            
        else:
            raise ValueError(f'Invalid objective: {self.objective}')
        
        new_qs = np.zeros(self.num_arms)
        for group, obj_vals in group_to_obj.items(): 
            new_qs[self.group_to_arm[group]] = obj_vals 
        self.update_q_values(new_qs)

    def update_q_values(self, new_q_values: np.ndarray):
        self.q_values = (1 - self.learning_rate) * self.q_values + self.learning_rate * new_q_values


class JointBandit(BanditPriority):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def extra_info_to_group(self, extra_info):
        return '-'.join([str(extra_info[feature]) for feature in self.feature])

    def data_to_group(self, data):
        if 'difficulty' in data['extra_info']:
            assert int(data['extra_info']['difficulty']) == float(data['extra_info']['difficulty']) # make sure difficulty is an integer
        return self.extra_info_to_group(data['extra_info'])


class BanditPPLPriority(JointBandit):
    """Bandit + PPL combined priority.

    Arm selection via Q-values (from JointBandit),
    within-arm ordering via PPL values.
    """

    def __init__(self, *args, **kwargs):
        self.dataset_size = kwargs.pop('dataset_size')
        super().__init__(*args, **kwargs)
        self.entropy_values = np.zeros(self.dataset_size)

    def __call__(self, data: dict[str, Any]):
        idx = data['extra_info'].get('index', 0)
        return self.entropy_values[idx]

    def update_entropy(self, entropy_dict: dict):
        for idx, entropy in entropy_dict.items():
            self.entropy_values[int(idx)] = entropy

    def get_metrics(self):
        bandit_metrics = {
            f'bandit/{group}_q_value': self.q_values[self.group_to_arm[group]]
            for group in self.group_to_arm
        }
        bandit_metrics.update({
            f'bandit/{group}_prob': self.get_probs(self.method)[self.group_to_arm[group]]
            for group in self.group_to_arm
        })
        valid = self.entropy_values[self.entropy_values != 0]
        if len(valid) > 0:
            bandit_metrics.update({
                'ppl/mean': float(np.mean(valid)),
                'ppl/std': float(np.std(valid)),
                'ppl/max': float(np.max(valid)),
                'ppl/min': float(np.min(valid)),
            })
        return bandit_metrics
